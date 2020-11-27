use bzip2::bufread::MultiBzDecoder;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::convert::TryInto;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom, BufRead, BufReader};
use std::path::Path;

/// File signature for BTree
const BTREE_MAGIC: [u8; 8] = *b"LOLBTREE";
/// Maximum number of branches per node
const BTREE_WIDTH: usize = 16;
/// The size of node in bytes
const NODE_SIZE: usize = core::mem::size_of::<BTreeNode>();
/// The size of header in bytes
const HEADER_SIZE: usize = core::mem::size_of::<BTreeHeader>();
/// Bytes size for chunks of nodes
const CHUNK_SIZE: usize = NODE_SIZE * BTREE_WIDTH;
/// How many bytes from key are going to be stored in the node.
const KEY_HEAD: usize = 8;

/// The on-disk representation of BTree is the following:
/// [header] [node][node][...] [string pool for keys]
///
/// Unless otherwise specified, each integer is u64.
///
/// [header] is [magic:u8*8][width:u64][size:u64][root offset:u64]
///
/// [node] is
/// [value:u64][branches:u32][key length:u32][key offset:u64][key head:u8*8]
///
/// For Branch nodes, branches is the number of child nodes and value is
/// the offset to the first child node.
/// For Leaf nodes, branches is zero, and value is the stored value
/// (in this particular case, offset in multistream bzip2 file).
///
/// [string pool] is just a continuous pool of bytes.  All of the information
/// necessary to interpret it is contained in nodes (key offset and length).
struct DiskBTree {
    file: File,
    length: u64,
    root_chunk: BTreeChunk,
    root_branches: usize,
}

impl DiskBTree {
    /// Open a BTree file
    pub fn open<P>(path: P) -> io::Result<Self>
        where P: AsRef<Path>
    {
        let mut file = File::open(path.as_ref())?;
        let mut head = [0; HEADER_SIZE];
        file.read_exact(&mut head)?;
        let header = BTreeHeader::from_buf(&head);
        if header.magic != BTREE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::Other, "invalid magic"));
        }
        if header.width as usize != BTREE_WIDTH {
            return Err(io::Error::new(io::ErrorKind::Other, "invalid width"));
        }
        let mut btree = Self {
            file,
            length: header.length,
            root_chunk: [BTreeNode::zero(); BTREE_WIDTH],
            root_branches: 0,
        };
        // make lint happy
        let _ = btree.length;

        let root = btree.read_chunk(header.root, 1)?[0];
        let root_chunk = btree.read_chunk(root.value, root.branches as usize)?;
        btree.root_chunk = root_chunk;
        btree.root_branches = root.branches as usize;
        Ok(btree)
    }

    /// Calculate the number of nodes and tree depth necessary
    /// to store the BTree.
    pub fn calc_size(mut length: u64) -> (u64, u64) {
        if length == 0 {
            return (0, 0)
        }
        let width = BTREE_WIDTH as u64;
        let mut nodes = 1;
        let mut depth = 0;
        while length > 1 {
            nodes += length;
            depth += 1;
            // calculate the number of parent nodes, rounding up
            length = (length - 1 + width) / width;
        }

        (nodes, depth)
    }

    /// Create a BTree file from a sorted collection of items.
    /// Providing a non-sorted collection will cause incorrect lookups.
    pub fn from_sorted<'a, P, I>(path: P, it: I) -> io::Result<Self>
        where P: AsRef<Path>,
              I: ExactSizeIterator<Item=(&'a str, u64)>
    {
        let length = it.len();
        let (nodes, _depth) = Self::calc_size(length as u64);
        let file = OpenOptions::new()
            .read(true).write(true).create(true)
            .open(path)?;
        let mut string_offset =
            (HEADER_SIZE as u64) + (NODE_SIZE as u64) * nodes;
        let mut out = [BTreeNode::zero(); BTREE_WIDTH];
        let mut btree = Self {
            file,
            root_chunk: out,
            length: length as u64,
            root_branches: 0,
        };

        let mut ii = 0;
        let mut node_offset = HEADER_SIZE as u64;
        let mut index = Vec::new();
        for (key, value) in it {
            let mut key_head = [0; KEY_HEAD];
            let head_len = key.len().min(key_head.len());
            let mut key_offset = 0;
            if key.len() > key_head.len() {
                assert!(key.len() < u32::max_value() as usize,
                        "Key is too big");
                btree.file.seek(SeekFrom::Start(string_offset))?;
                btree.file.write_all(&key.as_bytes()[8..])?;
                key_offset = string_offset;
                string_offset += (key.len() - 8) as u64;
            }
            key_head[..head_len].copy_from_slice(&key.as_bytes()[..head_len]);
            out[ii % BTREE_WIDTH] = BTreeNode {
                value,
                branches: 0,
                key_length: key.len() as u32,
                key_offset,
                key_head,
            };
            ii += 1;
            if ii % BTREE_WIDTH == 0 {
                btree.file.seek(SeekFrom::Start(node_offset))?;
                btree.write_chunk(&out)?;
                let node = out[BTREE_WIDTH - 1];
                index.push(node.to_index(BTREE_WIDTH as u32, node_offset));
                node_offset += (BTREE_WIDTH * NODE_SIZE) as u64;
            }
        }

        if ii % BTREE_WIDTH != 0 {
            btree.file.seek(SeekFrom::Start(node_offset))?;
            let len = ii % BTREE_WIDTH;
            btree.write_chunk(&out[..len])?;
            let node = out[len - 1];
            index.push(node.to_index(len as u32, node_offset));
            node_offset += (len * NODE_SIZE) as u64;
        }

        // let mut level = 0;
        // reduce index until we have one node.
        while index.len() > 1 {
            // level += 1;
            // println!("level {} len {}", level, index.len());
            let mut prev_index = Vec::new();
            core::mem::swap(&mut index, &mut prev_index);
            let mut ii = 0;
            for node in prev_index.drain(..) {
                out[ii % BTREE_WIDTH] = node;
                ii += 1;

                if ii % BTREE_WIDTH == 0 {
                    btree.write_chunk(&out)?;
                    let node = out[BTREE_WIDTH - 1];
                    index.push(node.to_index(BTREE_WIDTH as u32, node_offset));
                    node_offset += (BTREE_WIDTH * NODE_SIZE) as u64;
                }
            }

            if ii % BTREE_WIDTH != 0 {
                let len = ii % BTREE_WIDTH;
                btree.write_chunk(&out[..len])?;
                let node = out[len - 1];
                index.push(node.to_index(len as u32, node_offset));
                node_offset += (len * NODE_SIZE) as u64;
            }
        }

        let root = index[0];
        let root_chunk = btree.read_chunk(root.value, root.branches as usize)?;
        btree.root_chunk = root_chunk;
        btree.root_branches = root.branches as usize;
        btree.write_chunk(&index[..1])?;

        let mut buf = [0; HEADER_SIZE];
        BTreeHeader::new(length as u64, node_offset).to_buf(&mut buf);
        btree.file.seek(SeekFrom::Start(0))?;
        btree.file.write_all(&buf)?;

        Ok(btree)
    }

    /// Compares node key with provided key
    fn cmp_key(&mut self, node: BTreeNode, key: &str)
               -> Option<Ordering>
    {
        let key = key.as_bytes();
        let (key_head, key_tail) =
            if key.len() > 8 {
                (&key[..8], &key[8..])
            } else {
                (&key[..], &[][..])
            };

        let head_size = (node.key_length as usize).min(KEY_HEAD);
        let nkey_head = &node.key_head[..head_size];
        let cmp = nkey_head.cmp(key_head);
        if cmp != Ordering::Equal {
            return Some(cmp);
        }

        if node.key_length as usize <= KEY_HEAD {
            if key_tail.is_empty() {
                return Some(Ordering::Equal);
            } else {
                return Some(Ordering::Less);
            }
        }

        if key_tail.is_empty() {
            Some(Ordering::Greater)
        } else {
            let mut nkey_tail =
                vec![0; (node.key_length as usize) - KEY_HEAD];
            self.file.seek(SeekFrom::Start(node.key_offset)).ok()?;
            self.file.read_exact(&mut nkey_tail[..]).ok()?;
            Some(nkey_tail[..].cmp(key_tail))
        }
    }

    /// Lookup a `key` in the BTree
    pub fn lookup(&mut self, key: &str) -> Option<u64> {
        let mut chunk = self.root_chunk;
        let mut len = self.root_branches;

        let mut branches = true;
        let mut start_index = 0;
        'outer: while branches {
            for ii in start_index..len {
                let node = chunk[ii];
                let cmp = self.cmp_key(node, key)?;
                if node.branches == 0 {
                    branches = false;
                    match cmp {
                        Ordering::Equal => {
                            return Some(node.value);
                        }
                        Ordering::Greater => {
                            return None;
                        }
                        Ordering::Less => {}
                    }
                } else {
                    match cmp {
                        Ordering::Equal | Ordering::Greater => {
                            chunk = self.read_chunk(
                                node.value, node.branches as usize).ok()?;
                            len = node.branches as usize;
                            // If the comparison is exact, we know that
                            // the key is located in the last node.
                            start_index = if cmp == Ordering::Equal {
                                (node.branches as usize) - 1
                            } else {
                                0
                            };

                            continue 'outer;
                        }
                        Ordering::Less => {}
                    }
                }
            }
        }

        None
    }

    fn write_chunk(&mut self, nodes: &[BTreeNode]) -> io::Result<()> {
        assert!(nodes.len() <= BTREE_WIDTH,
                "too many nodes to write");
        let mut buf = [0u8; CHUNK_SIZE];
        for (chunk, node) in buf.chunks_exact_mut(NODE_SIZE).zip(nodes.iter()) {
            node.to_buf(chunk);
        }
        self.file.write_all(&buf[..NODE_SIZE * nodes.len()])?;
        Ok(())
    }

    /// Read a chunk of `BTreeNode`s
    fn read_chunk(&mut self, offset: u64, length: usize)
        -> io::Result<BTreeChunk>
    {
        assert!(length <= BTREE_WIDTH,
                "Exceeded BTree width chunk size in read_chunk");
        let mut buf = [0u8; CHUNK_SIZE];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut buf[..length * NODE_SIZE])?;
        let mut out = [BTreeNode::zero(); BTREE_WIDTH];
        for (chunk, node) in buf.chunks_exact(NODE_SIZE).zip(out.iter_mut()) {
            *node = BTreeNode::from_buf(chunk);
        }
        Ok(out)
    }
}

struct BTreeHeader {
    /// Magic signature for the BTree file
    magic: [u8; 8],
    /// Number of nodes in this BTree chunk
    width: u64,
    /// Number of nodes in this BTree
    length: u64,
    /// Offset to the root BTree node
    root: u64,
}

impl BTreeHeader {
    fn new(length: u64, root: u64) -> Self {
        Self {
            magic: BTREE_MAGIC,
            width: BTREE_WIDTH as u64,
            length,
            root,
        }
    }
    fn from_buf(buf: &[u8]) -> Self {
        assert!(buf.len() >= core::mem::size_of::<Self>(),
                "Not enough data to create header");
        let magic = buf[..8].try_into().unwrap();
        let width = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let length = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let root = u64::from_le_bytes(buf[24..32].try_into().unwrap());
        Self { magic, width, length, root }
    }
    fn to_buf(&self, buf: &mut [u8]) {
        assert!(buf.len() >= core::mem::size_of::<Self>(),
                "Not enough space to write header");
        buf[..8].copy_from_slice(&self.magic);
        buf[8..16].copy_from_slice(&self.width.to_le_bytes());
        buf[16..24].copy_from_slice(&self.length.to_le_bytes());
        buf[24..32].copy_from_slice(&self.root.to_le_bytes());
    }
}

#[derive(Clone, Copy)]
struct BTreeNode {
    /// Either value stored in node, or offset to child chunk
    value: u64,
    /// Number of child nodes (zero if this is a leaf)
    branches: u32,
    /// Length of key
    key_length: u32,
    /// Offset to key string
    key_offset: u64,
    /// Up to 8 first bytes of key, zero-padded
    key_head: [u8; KEY_HEAD],
}

impl BTreeNode {
    fn zero() -> Self {
        Self {
            value: 0,
            branches: 0,
            key_length: 0,
            key_offset: 0,
            key_head: [0; KEY_HEAD],
        }
    }
    fn to_buf(&self, buf: &mut [u8]) {
        assert!(buf.len() >= NODE_SIZE, "provided incorrect node size");
        buf[..8].copy_from_slice(&self.value.to_le_bytes());
        buf[8..12].copy_from_slice(&self.branches.to_le_bytes());
        buf[12..16].copy_from_slice(&self.key_length.to_le_bytes());
        buf[16..24].copy_from_slice(&self.key_offset.to_le_bytes());
        buf[24..32].copy_from_slice(&self.key_head);
    }
    fn from_buf(buf: &[u8]) -> Self {
        assert!(buf.len() == NODE_SIZE, "provided incorrect node size");
        let value = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let branches = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let key_length = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let key_offset = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let key_head = buf[24..32].try_into().unwrap();
        Self {
            value,
            branches,
            key_length,
            key_offset,
            key_head,
        }
    }
    fn to_index(&self, branches: u32, offset: u64) -> Self {
        Self {
            value: offset,
            branches,
            ..*self
        }
    }
}

type BTreeChunk = [BTreeNode; BTREE_WIDTH];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv = std::env::args().collect::<Vec<String>>();

    let input;
    if argv.len() < 2 {
        println!("please specify wiki index file name as an arguments");
        return Ok(());
    } else {
        input = &argv[1];
    }

    let index_path = Path::new(input).with_extension("bindex");

    let mut index;
    if index_path.is_file() {
        index = DiskBTree::open(index_path)?;
    } else {
        println!("Index not found, rebuilding...");
        let bufreader = BufReader::new(File::open(input)?);
        let input = MultiBzDecoder::new(bufreader);
        let input = BufReader::new(input);

        let start = std::time::Instant::now();
        let mut map = BTreeMap::new();

        for line in input.lines() {
            let line = line?;
            let mut iter = line.splitn(3, ':');
            let offset = iter.next().ok_or("No offset?")?
                .parse::<u64>().map_err(|_| "Offset is not a number")?;
            let _id = iter.next().ok_or("No page id?")?;
            let name: &str = iter.next().ok_or("No name?")?;
            map.insert(name.to_string(), offset);
        }
        println!("loaded dump in: {:?}", start.elapsed());

        let start = std::time::Instant::now();
        index = DiskBTree::from_sorted(
            index_path, map.iter().map(|(s, &v)| (s.as_str(), v)))?;
        println!("built btree in: {:?}", start.elapsed());
    }

    let mut buf = String::new();
    loop {
        buf.clear();
        let stdin = std::io::stdin();
        stdin.read_line(&mut buf)?;
        let key = buf.trim();
        if key.is_empty() { break; }
        let start = std::time::Instant::now();
        let value = index.lookup(key);
        println!("{:?} {:?}", value, start.elapsed());
    }

    Ok(())
}
