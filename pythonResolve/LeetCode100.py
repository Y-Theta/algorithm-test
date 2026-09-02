from Common import ListNode,TreeNode
from sortedcontainers import SortedList
from collections import deque

class Solution:

    def twoSum(self, nums: list[int], target: int) -> list[int]:
        aimset = dict()
        i = 0
        while i < len(nums):
            if nums[i] in aimset:
                return [aimset[nums[i]], i]
            aimset[(target - nums[i])] = i
            i += 1
        return []

    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        aimset = dict()
        for s in strs:
            key = "".join(sorted(s))
            if key not in aimset:
                aimset[key] = []
            aimset[key].append(s)
        return list(aimset.values())

    def longestConsecutive(self, nums: list[int]) -> int:
        if len(nums) == 0:
            return 0
        numset = set(nums)
        result = 0
        for i in numset:
            if (i - 1) in numset:
                continue
            scount = 1
            j = 1
            while i + j in numset:
                scount += 1
                j += 1
            result = max(result, scount)
        return result

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        start = 0
        offset = 1

        while True:
            if start + offset >= len(nums):
                break
            if nums[start] == 0:
                if nums[start + offset] == 0:
                    offset += 1
                else:
                    c = nums[start + offset]
                    nums[start] = c
                    nums[start + offset] = 0
                    start += 1
            else:
                start += 1

    def maxArea(self, height: List[int]) -> int:
        start = 0
        end = len(height) - 1

        result = 0
        while start < end:
            mheight = min(height[start], height[end])
            result = max(result, mheight * (end - start))
            if mheight == height[start]:
                start += 1
            elif mheight == height[end]:
                end -= 1
        return result

    def threeSum(self, nums: list[int]) -> list[list[int]]:
        sumdict: dict[int, list[int]] = dict()
        for i in range(len(nums)):
            if nums[i] not in sumdict:
                sumdict[nums[i]] = []
            sumdict[nums[i]].append(i)
        dictlist = list(sumdict)

    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        start = 0
        charset = [-1] * 256
        charset[ord(s[0])] = 0
        maxlen = 0
        for i in range(1, len(s)):
            cpos = ord(s[i])
            if charset[cpos] >= 0:
                maxlen = max(maxlen, i - start)
                start = max(start, charset[cpos] + 1)
            charset[cpos] = i
        maxlen = max(maxlen, len(s) - start)
        return maxlen

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals = sorted(intervals, key=lambda x:x[0])
        curend = intervals[0][1]
        cstart = intervals[0][0]
        result = []
        for i in range(1, len(intervals)):
            if intervals[i][0] <= curend:
                curend = max(curend, intervals[i][1])
            else:
                result.append([cstart, curend])
                cstart = intervals[i][0]
                curend = intervals[i][1]
        result.append([cstart, curend])
        return result

    def rotate(self, nums: list[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        lenn = len(nums)
        k = k % lenn
        if k == 0:
            return
        post = nums[0:-k]
        pre = nums[-k:lenn]
        nums[0:k] = pre
        nums[k:lenn] = post
    
    def maxSubArray(self, nums: List[int]) -> int:
        prefix = [0] * len(nums)
        prefix[0] = nums[0]
        result = prefix[0]
        minprefix = prefix[0]
        for i in range(1, len(nums)):
            prefix[i] = prefix[i - 1] + nums[i]
            if nums[i] > 0:
                result = max(result, prefix[i]- minprefix)
            minprefix = min(minprefix, prefix[i])
            result = max(result, prefix[i])
        
        return result

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        result = []
        rn = len(matrix)
        cn = len(matrix[0])
        level = min(rn,cn) // 2 + 1
        offset = 0
        xoffset = 1
        yoffset = 1
        while True:
            if offset > cn - offset - xoffset or offset > rn - offset - yoffset:
                break
            if offset == cn - offset - xoffset:
                for i in range(offset, rn - offset):
                    result.append(matrix[i][cn - offset - 1])
                break
            if offset == rn - offset - yoffset:
                for i in range(offset, cn - offset):
                    result.append(matrix[offset][i])
                break
            for i in range(offset, cn - offset - xoffset):
                result.append(matrix[offset][i])
            for i in range(offset, rn - offset - yoffset):
                result.append(matrix[i][cn - offset - 1])
            for i in range(offset, cn - offset - xoffset):
                result.append(matrix[rn - offset - 1][cn - i - 1])
            for i in range(offset, rn - offset - yoffset):
                result.append(matrix[rn - 1 - i][offset])
            offset += 1
            if offset > level:
                break

        return result

    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        fast = head
        pre = None
        originhead = head
        newhead = None

        if head.next == None:
            return True
        if newhead == None:
            return head.val == head.next.val

        while fast.next != None and fast.next.next != None:
            fast = fast.next.next
            newhead = head.next
            head.next = pre
            pre = head
            head = newhead

        newhead = newhead.next
        if fast.next != None:
            head.next = pre
            pre = head

        while newhead != None:
            if newhead.val != pre.val:
                return False
            newhead = newhead.next
            pre = pre.next
        return True

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head == None:
            return False
        fast = head
        while fast.next != None and fast.next.next != None:
            head = head.next
            fast = fast.next.next
            if head is fast:
                return True
        return False

    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        nodeset = set()
        while head != None:
            if head in nodeset:
                return head
            nodeset.add(head)
            head = head.next
        return None

    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        nodes = []
        while head != None:
            nodes.append(head)
            head = head.next
        if n == len(nodes):
            return head.next
        nodes[-(n + 1)].next = nodes[-n].next
        return head

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list1 == None:
            return list2
        elif list2 == None:
            return list1

        pre = None
        head = None
        while list1 != None or list2 != None:
            if list1 is not None:
                if list2 is not None:
                    if list1.val > list2.val:
                        if pre != None:
                            pre.next = list2
                            pre = pre.next
                        else:
                            pre = list2
                            originpre = pre
                        list2 = list2.next
                        pre.next = None
                    else:
                        if pre != None:
                            pre.next = list1
                            pre = pre.next
                        else:
                            pre = list1
                            originpre = pre
                        list1 = list1.next
                        pre.next = None
                else:
                    if pre != None:
                        pre.next = list1
                    else:
                        pre = list1
                        originpre = pre
                    break
            elif list2 is not None:
                if pre != None:
                    pre.next = list2
                else:
                    pre = list2
                    originpre = pre
                break
        return originpre

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        base = 0
        newval = None
        originval = None
        while l1 != None or l2 != None:
            cur = base
            if l1 != None:
                cur += l1.val
            if l2 != None:
                cur += l2.val
            base = cur // 10
            cur = cur % 10

            curnode = None
            if l1 != None:
                l1.val = cur
                curnode = l1
                l1 = l1.next
            if l2 != None:
                l2.val = cur
                curnode = l2
                l2 = l2.next

            if newval == None:
                newval = curnode
                originval = newval
            else:
                newval.next = curnode
                newval = newval.next
        if base > 0:
            newval.next = ListNode(1)
        return originval

    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        originhead = head
        if head == None:
            return None
        if head.next == None:
            return head
        originhead = head.next
        pre = None
        nextn = None
        while head != None and head.next != None:
            nextn = head.next.next
            if pre == None:
                pre = head.next
            else:
                pre.next = head.next
                pre = pre.next
            pre.next = head
            pre = pre.next
            pre.next = None
            head = nextn
        if nextn != None and pre != None:
            pre.next = nextn
            
        return originhead

    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if head == None:
            return None
        
        nlist = dict()
        index = 0
        newlist = []
        pre = None
        originhead = None
        while head != None:
            nlist[head] = index
            newnode = Node(head.val)
            newnode.random = head.random
            newlist.append(newnode)
            if pre == None:
                pre = newnode
                originhead = newnode
            else:
                pre.next = newnode
                pre = pre.next
            head = head.next
            index += 1
        
        head = originhead
        while head != None:
            if head.random != None and head.random in nlist:
                head.random = newlist[nlist[head.random]]
            head = head.next

        return originhead

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None:
            return None
        nodelist = []
        while head != None:
            nodelist.append(head)
            head = head.next
        
        nodelist.sort(key= lambda x :x.val)
        
        for i in range(len(nodelist)):
            if i + 1 < len(nodelist):
                nodelist[i].next = nodelist[i + 1]
            else:
                nodelist[i].next = None
        return nodelist[0]

    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if lists == None:
            return None
        listn = len(lists)
        if listn == 0:
            return None

        heap = lists
        def buildheap(index:int):
            while index >= 0:
                parent = (index + 1) // 2 - 1
                if parent < 0:
                    break
                if heap[index] == None:
                    break
                if heap[parent] == None or heap[parent].val > heap[index].val:
                    node = heap[parent]
                    heap[parent] = heap[index]
                    heap[index] = node
                    index = parent
                else:
                    break
        def alterNode(index:int):
            leftval, rightval = 10 ** 4 + 1,10 ** 4 + 1
            if (index + 1) * 2 < listn and heap[(index + 1) * 2] != None:
                rightval = heap[(index + 1) * 2].val
            if (index + 1) * 2 - 1 < listn and heap[(index + 1) * 2 - 1] != None:
                leftval = heap[(index + 1) * 2 - 1].val
            c = None
            if leftval > rightval:
                c = (index + 1) * 2
            else:
                c = (index + 1) * 2 - 1
            if c >= listn or heap[c] == None:
                return

            if  heap[index] == None or heap[c].val < heap[index].val:
                originnode= heap[index]
                heap[index] = heap[c]
                heap[c] = originnode
            alterNode(c)

        for i in range(len(heap)):
            buildheap(i)
        
        resulthead = None
        result = None
        while heap[0] != None:
            newnode = ListNode(heap[0].val)
            if resulthead == None:
                result = newnode
                resulthead = newnode
            else:
                result.next = newnode
                result = result.next
            heap[0] = heap[0].next
            alterNode(0)
        
        return resulthead

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root == None:
            return None
        originleft = root.left
        root.left = self.invertTree(root.right)
        root.right = self.invertTree(originleft)
        return root

    def isEqualTree(self, ta: Optional[TreeNode], tb: Optional[TreeNode]):
        if ta == None and tb == None:
            return True
        if ta != None and tb != None and ta.val == tb.val:
            return self.isEqualTree(ta.right, tb.left) and self.isEqualTree(ta.left, tb.right) 
        return False

    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root == None:
            return True
        return self.isEqualTree(root.left, root.right)

    maxval = 0
    def treeheight(self, root:Optional[TreeNode]) -> int:
        if root == None:
            return 0
        leftval = self.treeheight(root.left)
        rightval = self.treeheight(root.right)
        self.maxval = max(self.maxval, leftval + rightval)
        return 1 + max(leftval, rightval)

    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        final = self.treeheight(root)
        return self.maxval

    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def buildTree(start:int,end:int) -> Optional[TreeNode]:
            if start > end:
                return None
            mid = (start + end) // 2
            root = TreeNode(nums[mid])
            root.left = buildTree(start, mid - 1)
            root.right = buildTree(mid + 1, end)
            return root
        return buildTree(0, len(nums) - 1)

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def isValidBSTroot(root: Optional[TreeNode], nrange:tuple[Optional[int], Optional[int]]) -> bool:
            if root == None:
                return True
            if nrange[0] != None and root.val <= nrange[0]:
                return False
            if nrange[1] != None and root.val >= nrange[1]:
                return False
            lrange = [nrange[0], root.val]
            if nrange[1] != None:
                lrange = [nrange[0], min(root.val, nrange[1])]
            rrange = [root.val, nrange[1]]
            if nrange[0] != None:
                rrange = [max(nrange[0], root.val), nrange[1]]
            return isValidBSTroot(root.left, lrange) and isValidBSTroot(root.right, rrange)

        return isValidBSTroot(root, [None,None])

    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root == None:
            return []
        level = [root]
        result = []
        while len(level) > 0:
            templevel = list(level)
            templist = []
            level.clear()
            for v in templevel:
                templist.append(v.val)
                if v.left != None:
                    level.append(v.left)
                if v.right != None:
                    level.append(v.right)
            result.append(templist)
        return result

    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root == None:
            return []
        level = [root]
        result = []
        while len(level) > 0:
            templevel = list(level)
            templist = []
            level.clear()
            for v in templevel:
                templist.append(v.val)
                if v.left != None:
                    level.append(v.left)
                if v.right != None:
                    level.append(v.right)
            if len(templist) > 0:
                result.append(templist[-1])
        return result

    def numIslands(self, grid: List[List[str]]) -> int:
        visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
        final = 0
        rc = len(grid)
        cc = len(grid[0])
        for r in range(rc):
            for c in range(cc):
                if visited[r][c]:
                    continue
                if grid[r][c] == '1':
                    final += 1
                    stack = []
                    stack.append((r,c))
                    visited[r][c] = True
                    while len(stack) > 0:
                        tempstack = list(stack)
                        stack.clear()
                        for j in tempstack:
                            if j[0] - 1 >= 0 and grid[j[0] - 1][j[1]] == '1' and not visited[j[0] - 1][j[1]]:
                                visited[j[0] - 1][j[1]] = True
                                stack.append((j[0] - 1, j[1]))
                            if j[0] + 1 < rc and grid[j[0] + 1][j[1]] == '1' and not visited[j[0] + 1][j[1]]:
                                visited[j[0] + 1][j[1]] = True
                                stack.append((j[0] + 1, j[1]))
                            if j[1] - 1 >= 0 and grid[j[0]][j[1] - 1] == '1' and not visited[j[0]][j[1] - 1]:
                                visited[j[0]][j[1] - 1] = True
                                stack.append((j[0], j[1] - 1))
                            if j[1] + 1 < cc and grid[j[0]][j[1] + 1] == '1' and not visited[j[0]][j[1] + 1]:
                                visited[j[0]][j[1] + 1] = True
                                stack.append((j[0], j[1] + 1))
        return final

    def orangesRotting(self, grid: List[List[int]]) -> int:
        rlen = len(grid)
        clen = len(grid[0])

        totalcount = 0
        badorigins = []
        for r in range(rlen):
            for c in range(clen):
                if grid[r][c] == 1:
                    totalcount += 1
                if grid[r][c] == 2:
                    badorigins.append((r,c))
        
        if totalcount == 0:
            return 0
        
        if len(badorigins) == 0:
            if totalcount > 0:
                return -1
            return 0
        
        time = 0
        while len(badorigins) > 0:
            tempbad = list(badorigins)
            badorigins.clear()
            for i in tempbad:
                y,x = i[0],i[1]
                if y - 1 >= 0 and grid[y - 1][x] == 1:
                    totalcount -= 1
                    grid[y- 1][x] = 2
                    badorigins.append((y - 1,x))
                if y + 1 < rlen and grid[y + 1][x] == 1:
                    totalcount -= 1
                    grid[y + 1][x] = 2
                    badorigins.append((y + 1,x))
                if x - 1 >= 0 and grid[y][x - 1] == 1:
                    totalcount -= 1
                    grid[y][x - 1] = 2
                    badorigins.append((y,x - 1))
                if x + 1 < clen and grid[y][x + 1] == 1:
                    totalcount -= 1
                    grid[y][x + 1] = 2
                    badorigins.append((y,x + 1))
            time += 1
        
        if totalcount > 0:
            return -1
        
        return time

    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        result = []

        def pickone(id:int = 0):
            if id == n:
                result.append(nums[:])
            
            for i in range(id,n):
                nums[i], nums[id] = nums[id], nums[i]
                pickone(id + 1)
                nums[i], nums[id] = nums[id], nums[i]
        pickone()
        return result

    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        temp = []

        def pickone(id:int = 0):
            if id == len(nums):
                result.append(temp[:])
                return
            
            temp.append(nums[id])
            pickone(id + 1)
            temp.pop(-1)
            pickone(id + 1)

        pickone()
        return result
    
    def letterCombinations(self, digits: str) -> List[str]:
        results = []
        temp = []

        numbers = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        def pickone(id:int = 0):
            if id == len(digits):
                results.append(''.join(temp))
                return
            
            mystr = numbers[digits[id]]
            for i in range(len(mystr)):
                temp.append(mystr[i])
                pickone(id + 1)
                temp.pop(-1)
        pickone()
        return results

    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root == None:
            pass
        def findleft(node:Optional[TreeNode]) -> Optional[TreeNode]:
            if node == None:
                return
            if node.left == None:
                return node
            originright = node.right
            originleft = node.left
            node.right = originleft
            lastnode = findleft(originleft)
            lastnode.right = originright
            return findleft(originright)
        findleft(root)

    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        predict = dict()
        freedict = dict()
        for p in prerequisites:
            if p[0] not in predict:
                predict[p[0]] = set()
            if p[1] not in freedict:
                freedict[p[1]] = set()
            freedict[p[1]].add(p[0])
            predict[p[0]].add(p[1])
        
        while True:
            flag = False
            for i in list(freedict):
                if i not in predict:
                    flag = True
                    for k in freedict[i]:
                        predict[k].discard(i)
                        if len(predict[k]) == 0:
                            predict.pop(k)
                    freedict.pop(i)
            if not flag:
                break
        
        return len(predict) == 0
    
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root == None:
            pass
        def findleft(node:Optional[TreeNode]) -> Optional[TreeNode]:
            if node == None:
                return
            if node.left == None:
                if node.right != None:
                    return findleft(node.right)
                return node
            originright = node.right
            originleft = node.left
            node.right = originleft
            node.left = None
            lastnode = findleft(originleft)
            lastnode.right = originright
            if originright == None:
                return lastnode
            return findleft(originright)
        findleft(root)

    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        def buildTreeRoot(rootid:int, start:int ,end:int) -> Optional[TreeNode]:
            if start > end:
                return None
            root = TreeNode(preorder[rootid])
            if start == end:
                return root
            rootpos = inorder.index(preorder[rootid])
            leftcount = rootpos - start
            root.left = buildTreeRoot(rootid + 1, start, rootpos - 1)
            root.right = buildTreeRoot(rootid + leftcount + 1, rootpos + 1, end)
            return root
        return buildTreeRoot(0, 0, len(inorder) - 1)

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        total = 0
        def dfs(root:Optional[TreeNode], avaliablesums:list[int]):
            nonlocal total
            if root == None:
                return
            if avaliablesums != None:
                for i in range(len(avaliablesums)):
                    avaliablesums[i] += root.val
                    if avaliablesums[i] == targetSum:
                        total += 1
            if root.val == targetSum:
                total += 1
            avaliablesums.append(root.val)
            if root.left != None:
                dfs(root.left, avaliablesums)
            if root.right != None:
                dfs(root.right, avaliablesums)
            avaliablesums.remove(root.val)
            for i in range(len(avaliablesums)):
                avaliablesums[i] -= root.val
        dfs(root, [])
        return total

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        aimnode = None
        def dfs(root:Optional[TreeNode], aim1:TreeNode, aim2:TreeNode) -> tuple[bool,bool]:
            nonlocal aimnode
            result = [False, False]
            if root == None:
                return result
            if root == aim1:
                result = [True, result[1]]
            if root == aim2:
                result = [result[0], True]
            leftval = dfs(root.left, aim1, aim2)
            rightval = dfs(root.right, aim1, aim2)
            result = [leftval[0] or rightval[0] or result[0], leftval[1] or rightval[1] or result[1]]
            if result == [True,True] and aimnode == None:
                aimnode = root
            return result
        dfs(root, p, q)
        return aimnode

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates = sorted(candidates)
        result = []

        def pickone(start:int, current: list[int], currentsum:int):
            for i in range(len(candidates)):
                if i < start:
                    continue
                if currentsum + candidates[i] > target:
                    break
                if currentsum + candidates[i] == target:
                    current.append(candidates[i])
                    result.append(current[:])
                    current.pop(-1)
                    break
                current.append(candidates[i])
                pickone(i ,current, currentsum + candidates[i])
                current.pop(-1)
        pickone(0, [], 0)
        return result

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        def bisect_left(target:int, row:list[int], start:int ,end:int) -> int:
            while start <= end:
                mid = (start + end) // 2
                if row[mid] > target:
                    end = mid - 1
                elif row[mid] == target:
                    return mid
                else:
                    start = mid + 1
            return (start + end) // 2 + 1
        def search(target:int , startr:int ,startc:int,endr:int,endc:int) -> bool:
            if startr >= len(matrix):
                return False
            if endc < 0:
                return False
            cid = bisect_left(target, matrix[startr], startc, endc)
            if cid > endc:
                return search(target, startr + 1, startc, endr, endc)
            if matrix[startr][cid] == target:
                return True
            return search(target, startr, startc, endr, cid - 1)
        return search(target, 0,0,len(matrix) - 1, len(matrix[0]) - 1)

    def generateParenthesis(self, n: int) -> List[str]:
        result = []
        finalresult = []
        def pickone(start:int = 0, picked:int = 0):
            if start == n and picked == 0:
                finalresult.append(''.join(result[:]))
            
            if start < n:
                result.append('(')
                pickone(start + 1, picked + 1)
                result.pop(-1)
            if picked > 0:
                result.append(')')
                pickone(start + 1, picked - 1)
                result.pop(-1)
        pickone()
        return finalresult
    
    def exist(self, board: List[List[str]], word: str) -> bool:
        visited = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]
        startpos = []
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == word[0]:
                    startpos.append((r,c))
        if len(startpos) == 0:
            return False
        def pickone(start:int ,r:int , c:int) -> bool:
            if start == len(word):
                return True
            if r < 0 or r >= len(board) or c < 0 or c > len(board[0]):
                return False
            if visited[r][c]:
                return False
            if board[r][c] == word[start]:
                visited[r][c] = True
                flag = False
                flag = flag or pickone(start + 1, r-1,c)
                flag = flag or pickone(start + 1, r+1,c)
                flag = flag or pickone(start + 1, r,c-1)
                flag = flag or pickone(start + 1, r,c+1)
                if flag:
                    return True
                visited[r][c] = False
        
        for i in range(len(startpos)):
            if pickone(0, i[0], i[1]):
                return True
            
        return False

    def partition(self, s: str) -> List[List[str]]:
        result = []
        finalresult = []

        def ispal(s:str) -> bool:
            for i in range(len(s) // 2):
                if s[i] != s[-i-1]:
                    return False
            return True

        def pickone(start:int = 0):
            nonlocal result
            if start == len(s):
                finalresult.append(result[:])
                return
            
            for i in range(start, len(s)):
                if ispal(s[start: i + 1]):
                    result.append(s[start: i + 1])
                    pickone(i)
                    result.pop(-1)
        pickone()
        return finalresult

    def isValid(self, s: str) -> bool:
        stack = []
        cmap = {')':'(' , ']':'[' , '}':'{'}
        for c in s:
            if c == '(' or  c == '[' or c == '{':
                stack.append(c)
            elif c == ')' or c == ']' or c == '}':
                if len(stack) == 0:
                    return False
                if stack[-1] == cmap[c]:
                    stack.pop(-1)
                else:
                    return False
        return len(stack) == 0

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        def biosearchcol(col:int, target:int, startr:int, endr:int) -> int:
            while startr <= endr:
                mid = (startr + endr) // 2
                val = matrix[mid][col]
                if val > target:
                    endr = mid - 1
                elif val == target:
                    return mid
                else:
                    startr = mid + 1
            return (startr + endr) // 2 + 1
        def biosearchrow(row:int, target:int, startc:int, endc:int) -> int:
            while startc <= endc:
                mid = (startc + endc) // 2
                val = matrix[row][mid]
                if val > target:
                    endc = mid - 1
                elif val == target:
                    return mid
                else:
                    startc = mid + 1
            return (startc + endc) // 2 + 1
        def search(tar:int, startr:int, startc:int, endr:int, endc:int) -> bool:
            if startr >= len(matrix):
                return False
            if endc < 0:
                return False
            
            cidx = biosearchrow(startr, target, startc, endc)
            if cidx > endc:
                ridx = biosearchcol(endc, target, startr, endr)
                if ridx > endr:
                    return False
                elif matrix[ridx][endc] == target:
                    return True
                return search(tar, startr, startc, ridx - 1, endc)
            else:
                if matrix[startr][cidx] == target:
                    return True
                return search(tar, startr, startc, endr, cidx - 1)

        return search(target, 0,0,len(matrix) - 1,len(matrix[0]) - 1)
    
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if nums == None or len(nums) == 0:
            return [-1,-1]
        def bisect_left_(target:int, start:int ,end:int):
            while start < end:
                mid = (start + end) // 2
                if nums[mid] > target:
                    end = mid - 1
                elif nums[mid] < target:
                    start = mid + 1
                else:
                    end = mid
            return (start + end) // 2
        def bisect_right_(target:int, start:int ,end:int):
            while start <= end:
                mid = (start + end) // 2
                if nums[mid] > target:
                    end = mid - 1
                else:
                    start = mid + 1
            return (start + end) // 2
        pos = bisect_left_(target,0 ,len(nums) - 1)
        if nums[pos] != target:
            return [-1,-1]
        pos1 = bisect_right_(target,0 ,len(nums) - 1)
        return [pos,pos1]
            
    def search(self, nums: List[int], target: int) -> int:
        def search_broken(start:int, end:int) -> int:
            while start <= end:
                mid = (start + end) // 2
                if nums[mid] >= nums[0]:
                    start = mid + 1
                else:
                    end = mid - 1
            return (start + end) // 2
        def search_item(tar:int, start:int, end:int) -> bool:
            while start <= end:
                mid = (start + end) // 2
                if nums[mid] > tar:
                    end = mid - 1
                elif nums[mid] < tar:
                    start = mid + 1
                else:
                    return mid
            return -1
        idx = search_broken(0, len(nums) - 1)
        if idx < 0:
            return search_item(target, 0, len(nums) - 1)
        if nums[idx] >= nums[0]:
            return max(search_item(target, 0, idx), search_item(target, idx + 1, len(nums) - 1))
        return max(search_item(target, 0, idx - 1), search_item(target, idx, len(nums) - 1))
    
    def decodeString(self, s: str) -> str:
        start = 0
        plusstack = []
        ord0 = ord('0')
        tempstr = ""
        for i,c in enumerate(s):
            if c == '[':
                plusstack.append((tempstr, int(s[start:i])))
                tempstr = ""
                continue
            elif c == ']':
                laststack = plusstack[-1]
                laststr = laststack[0]
                repeattimes = laststack[1]
                laststack.pop(-1)
                for i in range(repeattimes):
                    laststr += tempstr
                tempstr = laststr
                continue
            if i > 0 and ord(s[i]) - ord0 < 10 and (ord(s[i-1]) - ord0 > 10 or ord(s[i-1])  - ord0< 0):
                start = i
            if ord(c) - ord('a') >= 0 and ord(c) - ord('a') < 26:
                tempstr += c 
        
        return tempstr
        
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        flag = False
        slist = SortedList()
        for i in range(len(nums) - 1, -1 ,-1):
            slist.add(nums[i])
            idx = slist.bisect_right(nums[i])
            if idx < len(slist):
                flag = True
                nums[i] = slist[idx]
                slist.pop(idx)
                for j in range(i + 1,len(nums)):
                    nums[j] = slist[0]
                    slist.pop(0)
                break
        if not flag:
            nums[0,len(nums)] = sorted(nums)
        
    def minDistance(self, word1: str, word2: str) -> int:
        lenw1 = len(word1) + 1
        lenw2 = len(word2) + 1
        dp = [[0 for _ in range(lenw2)] for _ in range(lenw1)]
        dp[0][0] = 0
        for i in range(1, lenw1):
            dp[i][0] = dp[i - 1][0] + 1
        for i in range(1, lenw2):
            dp[0][i] = dp[0][i - 1] + 1

        for i in range(1, lenw1):
            for j in range(1, lenw2):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[lenw1 - 1][lenw2 - 1]
    
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        answer = [0] * len(temperatures)
        for i,v in enumerate(temperatures):
            if len(stack) > 0 and stack[-1][1] < v:
                while len(stack) > 0 and stack[-1][1] < v:
                    top = stack.pop(-1)
                    answer[top[0]] = i - top[0]
            stack.append((i,v))
        return answer

    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        dp = [0] * n
        dp[0] = 0
        stack = deque()
        if s[0] == '(':
            stack.append(0)
        maxval = 0

        for i in range(1, len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                if len(stack) > 0 and s[stack[-1]] == '(':
                    stack.pop()
                    if len(stack) == 0:
                        maxval = max(maxval, i + 1)
                    else:
                        maxval = max(maxval, i - stack[-1])
                else:
                    stack.append(i)
        return maxval

    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = deque()
        maxval = 0
        n = len(heights)
        for i in range(n + 1):
            while len(stack) > 0 and (i >=n or heights[i] < stack[-1][1]):
                top = stack.pop()
                if len(stack) == 0:
                    maxval = max(maxval, top[1] * i)
                else:
                    maxval = max(maxval, top[1] * (i - stack[-1][0] - 1))
            if i < n:
                stack.append((i, heights[i]))
        return maxval

    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        elif n == 2:
            return 2
        dp = [0] * n
        dp[0] = 1
        dp[1] = 2

        for i in range(2, n):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n -1]

    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        elif len(nums) == 2:
            return max(nums[0],nums[1])

        dp = [0] * len(nums)
        dp[0] = nums[0]
        dp[1] = max(nums[1],nums[0])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-2] + nums[i], dp[i-1])
        
        return dp[len(nums) - 1]

    def numSquares(self, n: int) -> int:
        if n == 1:
            return 1

        dp = [0] * (n + 1)
        dp[0] = 0
        dp[1] = 1

        nums = [i * i for i in range(1, 101)]
        
        for i in range(2, n + 1):
            dp[i] = 10 ** 4
            for j in range(len(nums)):
                if nums[j] > i:
                    break
                dp[i] = min(dp[i], 1 + dp[i - nums[j]])
        
        return dp[n]

    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0

        maxlen = amount+1
        dp = [-1] * maxlen
        dp[0] = 0

        for i in coins:
            if i > len(dp):
                break
            dp[i] = 1

        for i in range(min(coins) + 1, maxlen):
            minval = maxlen
            for j in coins:
                if i - j >= 0 and dp[i - j] > 0: 
                    minval = min(minval, dp[i - j] + 1)
            if minval >= 0 and minval != maxlen:
                dp[i] = minval
        
        return dp[amount]

    def canPartition(self, nums: List[int]) -> bool:
        half = sum(nums) // 2
        sumset = set()

        for i in range(len(nums)):
            for j in list(sumset):
                if j + nums[i] not in sumset:
                    sumset.add(j + nums[i])
            if nums[i] not in sumset:
                sumset.add(nums[i])
            if half in sumset:
                return True
        return False
