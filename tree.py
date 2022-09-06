class Node:
    def __init__(self, value = None, lvalue = None, rvalue = None):
        self.value = value
        self.left = lvalue
        self.right = rvalue

class Tree:
    def __init__(self, root = None):
        self.root = root

    def push(self, value):
        node = Node(value=value)
        tempNode = self.root

        if self.root is None:
            self.root = node
            return
        else:
            ptrNode = self.root
            while ptrNode is not None:
                tempNode = ptrNode
                if node.value < ptrNode.value:
                    ptrNode = ptrNode.left
                else:
                    ptrNode = ptrNode.right

            if node.value < tempNode.value:
                tempNode.left = node
            else:
                tempNode.right = node

    def removeNode(self, value):
            temp = self.root
            self.pop(temp, value)

    def pop(self, node, value):
        if node is None:
            return -1
        elif node.value > value:
            node.left = self.pop(node.left, value)
        elif node.value < value:
            node.right = self.pop(node.right, value)
        else:
            temp = node
            if node.right is None and node.left is None:
                del node
                node = None
            elif node.right is None:
                node = node.left
                del temp
            elif node.left is None:
                node = node.right
                del temp
            else:
                temp = self.searchMaxNode(node.left)
                node.value = temp.value
                node.left = self.pop(node.left, temp.value)
        return node

    def searchMaxNode(self):
            if self.root is None:
                return
            else:
                temp = self.root
                maxVal = 0
                while temp is not None:
                    maxVal = temp.value
                    temp = temp.right
                print("Max Value : ", maxVal)
                return 0

    def searchNode(self, value):
        temp = self.root
        tempRoot = Node()
        while temp is not None:
            if temp.value == value:
                print("Find Value")
                return True
            elif temp.value > value:
                temp = temp.left
            else:
                temp = temp.right
        print("Value Not Found")
        return False

    def show(self):
            temp = self.root
            print("What kind of Traverse(1.inorder, 2.preodrer, 3.postorder")
            a = int(input())
            if a == 1:
                self.inOrder(temp)
            elif a == 2:
                self.preOrder(temp)
            elif a == 3:
                self.postOrder(temp)
            else:
                return -1

    def inOrder(self, node):
        if node is not None:
            self.inOrder(node.left)
            print(node.value)
            self.inOrder(node.right)

    def preOrder(self, node):
        if node is not None:
            print(node.value)
            self.preOrder(node.left)
            self.preOrder(node.right)

    def postOrder(self, node):
        if node is not None:
            self.postOrder(node.left)
            self.postOrder(node.right)
            print(node.value)

if __name__ == "__main__":
    tree = Tree()
    tree.push(100)
    tree.push(40)
    tree.push(50)
    tree.push(10)
    tree.push(15)
    tree.push(30)
    tree.push(20)
    tree.push(50)
    tree.push(40)
    tree.show()
    tree.searchMaxNode()