from Common import ListNode,TreeNode

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
