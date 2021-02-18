# DiskBTree

This is a quick and dirty imlpementation of on-disk read-only BTree.
The purpose of this project is to build a fast and self-contained index for
Wikipedia's [multistream-index.txt files](https://dumps.wikimedia.org/tpiwiktionary/), instead of using a full database.

The on-disk representation of BTree is the following:

`[header] [node][node][...] [string pool for keys]`

`[header]` is `[magic:u8*8][width:u64][size:u64][root_offset:u64]`
`root_offset` points to the root node.

`[node]` is
`[value:u64][branches:u32][key_length:u32][key_offset:u64][key_head:u8*8]`

For Branch nodes, branches is the number of child nodes and value is
the offset to the first child node.

For Leaf nodes, branches is zero, and value is the stored value
(in this particular case, offset in multistream bzip2 file).

For Branch nodes, key represents the highest key of all of its children,
for Leaf nodes, key corresponds to the node itself.

`[string pool]` is simply a continuous pool of bytes.  All of the information
necessary to interpret it is contained in nodes (key offset and length).
String pool only contains the tail bytes of a key.
If a key fits in node field `key_head`, string pool is not used.

# License

The MIT License
