from Common import SegmentTreeNode

root: SegmentTreeNode = SegmentTreeNode(0, 0, SegmentTreeNode.minoperation)
numlist = [1, 3, 4, 5, 6, 6, 3, 1, 12, 3, 4, 5, 23, 2, 3, 2, 2, 1]
root.start = 0
root.end = len(numlist) - 1
SegmentTreeNode.build_segment_tree(numlist, root, 0, len(numlist) - 1)
None