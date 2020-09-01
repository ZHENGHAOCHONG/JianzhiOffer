1.在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。（已看视频）
方法一：暴力法。逐个遍历，时间复杂度O（n2）
public class Solution {
    public boolean Find(int target, int [][] array) {
        for(int i=0;i<array.length;i++){//数组名.length 就是数组的行，有多少行
            for(int j=0;j<array[0].length;j++){ //二维矩阵中的第一行array[0]的列数，有多少列
                if(array[i][j] == target){
                    return true;
                }
            }
        }
        return false;
    }
}
方法二：从左下找，时间复杂度O（n），右上同理
思路：利用该二维数组的性质：
每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序
改变个说法，即对于左下角的值 m，m 是该行最小的数，是该列最大的数
每次将 m 和目标值 target 比较：
当 m < target，由于 m 已经是该行最大的元素，想要更大只有从列考虑，取值右移一位
当 m > target，由于 m 已经是该列最小的元素，想要更小只有从行考虑，取值上移一位
当 m = target，找到该值，返回 true
用某行最小或某列最大与 target 比较，每次可剔除一整行或一整列
public class Solution {
    public boolean Find(int target, int [][] array) {
        int rows = array.length;
        if(rows == 0){ //特例情况
            return false;
        }
        int cols = array[0].length;
        if(cols == 0){
            return false;
        }
        // 左下
        int row = rows-1;
        int col = 0;
        while(row>=0 && col<cols){
            if(array[row][col] < target){ //（m，n）小于target，列值增加
                col++;
            }else if(array[row][col] > target){
                row--;
            }else{ //（m，n）等于target
                return true;
            }
        }
        return false;
    }
}
2.请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
方法一：调用自带函数
public class Solution {
    public String replaceSpace(StringBuffer str) {
        return str.toString().replace(" ", "%20");
    }
}
方法二：用新的数组存
import java.util.*;
public class Solution {
    public String replaceSpace(StringBuffer str) {
        StringBuilder sb = new StringBuilder();
        for(int i=0;i<str.length();i++){
            char c = str.charAt(i);
            if(c == ' '){
                sb.append("%20");
            }else{
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
*****3.输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
方法一：递归
import java.util.*;
public class Solution {
    ArrayList<Integer> list = new ArrayList(); //新建数组列表
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        if(listNode!=null){ //判定
            printListFromTailToHead(listNode.next);
            list.add(listNode.val);
        }
        return list;
    }
}
*****4.输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
方法：递归构建二叉树
/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
import java.util.Arrays;
public class Solution {
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        if (pre.length == 0 || in.length == 0) { //特例情况
            return null;
        }
        TreeNode root = new TreeNode(pre[0]); //新建二叉树组，定义root为前序第一位
        // 在中序中找到前序的根，遍历中序
        for (int i = 0; i < in.length; i++) {
            if (in[i] == pre[0]) {//和前序第一位相等
                // 左子树，注意 copyOfRange 函数，左闭右开
                root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
                // 右子树，注意 copyOfRange 函数，左闭右开
                root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length), Arrays.copyOfRange(in, i + 1, in.length));
                break;
            }
        }
        return root;
    }
}
思路：根据中序遍历和前序遍历可以确定二叉树，具体过程为：
根据前序序列第一个结点确定根结点 ；根据根结点在中序序列中的位置分割出左右两个子序列 ；对左子树和右子树分别递归使用同样的方法继续分解 
例如：
前序序列{1,2,4,7,3,5,6,8} = pre
中序序列{4,7,2,1,5,3,8,6} = in
根据当前前序序列的第一个结点确定根结点，为 1 
找到 1 在中序遍历序列中的位置，为 in[3] 
切割左右子树，则 in[3] 前面的为左子树， in[3] 后面的为右子树 
则切割后的左子树前序序列为：{2,4,7}，切割后的左子树中序序列为：{4,7,2}；切割后的右子树前序序列为：{3,5,6,8}，切割后的右子树中序序列为：{5,3,8,6} 
对子树分别使用同样的方法分解
5.用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
import java.util.Stack;
public class Solution {
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
 
    public void push(int node) {
        stack1.push(node);
    }
 
    public int pop() {
        if (stack2.size() <= 0) { //判断2是否为null
            while (stack1.size() != 0) {  //1中存在元素
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop();
    }
}
思路：队列的特性是：“先入先出”，栈的特性是：“先入后出”
当我们向模拟的队列插入数 a,b,c 时，假设插入的是 stack1，此时的栈情况为：
栈 stack1：{a,b,c}      栈 stack2：{} 
当需要弹出一个数，根据队列的"先进先出"原则，a 先进入，则 a 应该先弹出。但是此时 a 在 stack1 的最下面，将 stack1 中全部元素逐个弹出压入 stack2，现在可以正确的从 stack2 中弹出 a，此时的栈情况为：若要弹出a，则需先将b和c放到另一个栈中。
栈 stack1：{}            栈 stack2：{c,b} 
继续弹出一个数，b 比 c 先进入"队列"，b 弹出，注意此时 b 在 stack2 的栈顶，可直接弹出，此时的栈情况为：
栈 stack1：{}            栈 stack2：{c} 
此时向模拟队列插入一个数 d，还是插入 stack1，此时的栈情况为：
栈 stack1：{d}           栈 stack2：{c} 
弹出一个数，c 比 d 先进入，c 弹出，注意此时 c 在 stack2 的栈顶，可直接弹出，此时的栈情况为：
栈 stack1：{d}           栈 stack2：{c} 
根据上述例子可得出结论：
当插入时，直接插入 stack1 ；
当弹出时，当 stack2 不为空，弹出 stack2 栈顶元素，如果 stack2 为空，将 stack1 中的全部数逐个出栈入栈 stack2，再弹出 stack2 栈顶元素。
*****6.把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
public class Solution {
    public int minNumberInRotateArray(int[] array) {
        int i = 0, j = array.length - 1;
        if (array[i] < array[j]) { // 2  返回小的那个元素
            return array[i];
        }
        if (array[i] == array[j] && array[i] == array[(i + j) >> 1]) { // 3  数组中有相等元素
            int min = array[i];
            for (; i <= j; i++) {  //遍历数组中所有元素，如果比相等的元素值要小的话，最小值就是新的array[i]
                if (array[i] < min) {
                    min = array[i];
                }
            }
            return min;
        }
        while (i + 1 < j) { // 1  这段有些没看懂 应该是二分法
            int mid = (i + j) >> 1; //找到二分法中点
            if (array[mid] >= array[i]) { //如果中点值比array[i]的值要大的话
                i = mid;
            } else if (array[mid] <= array[j]) { //中点值比array[i]的值要小且又小于等于array[j]的话
                j = mid;
            }
        }
        return array[j];
    }
}
**7.大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。n<=39
public class Solution {
    public int Fibonacci(int n) {
        int ans[] = new int[40]; //新建一个数组ans
        ans[0] = 0;  //指定数组的前两位为0和1
        ans[1] = 1;
        for(int i=2;i<=n;i++){ //从第三位开始遍历整个数组
            ans[i] = ans[i-1] + ans[i-2]; //这句有些不懂，意思就是第三位是前两位数字之和
        }
        return ans[n];
    }
}
方法：优化递归法
8.一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
思路：跳n级台阶相当于n-1和n-2级台阶的和；跳一级台阶有1种方法；跳两级台阶有2种方法；跳三级台阶有三种方法；跳四级台阶有5种方法；以此类推。。。所以跳n级台阶相当于n-1和n-2级台阶的和
原因：n级台阶就相当于n-1级再跳一次一阶的和n-2级再跳一次2阶的，即斐波那契数列
public class Solution { 
    public int JumpFloor(int target) {
        if (target==1 || target==2){ //如果台阶数是1或者2就返回target
            return target;
        }
        else{  //大于2的时候返回
            return JumpFloor(target-1)+JumpFloor(target-2);
        }
    }
}
*****9.一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
思路：易知 f(n)=f(n-1)+f(n-2)+……f(1)   f(n-1)=f(n-2)+……f(1)
两式相减得f(n)=2f(n-1)
public class Solution {
    public int JumpFloorII(int target) {
        return 1<<(target-1);
        //return (int)Math.pow(2,target-1); //这句话不懂，这句极是返回2f(n-1)
    }
}
**法二：
public class Solution {
    public int JumpFloorII(int target) {
        if (target <= 0) { 
            return -1;
        } else if (target == 1) { //判断两个特例情况
            return 1;
        } else {  //否则的话就返回2f(n-1)
            return 2 * JumpFloorII(target - 1);
        }
    }
}
思路：  
1）这里的f(n) 代表的是n个台阶有一次1,2,...n阶的 跳法数。 
2）n = 1时，只有1种跳法，f(1) = 1 
3) n = 2时，会有两个跳得方式，一次1阶或者2阶，这回归到了问题（1） ，f(2) = f(2-1) + f(2-2)  
4) n = 3时，会有三种跳得方式，1阶、2阶、3阶， f(3)=f(3-1)+f(3-2)+f(3-3)
    那么就是第一次跳出1阶后面剩下：f(3-1);第一次跳出2阶，剩下f(3-2)；第一次3阶，那么剩下f(3-3) 
    因此结论是f(3) = f(3-1)+f(3-2)+f(3-3) 
5) n = n时，会有n中跳的方式，1阶、2阶...n阶，得出结论： 
    f(n) = f(n-1)+f(n-2)+...+f(n-(n-1)) + f(n-n) => f(0) + f(1) + f(2) + f(3) + ... + f(n-1)
6) 由以上已经是一种结论，但是为了简单，我们可以继续简化： 
    f(n-1) = f(0) + f(1)+f(2)+f(3) + ... + f((n-1)-1) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) 
    f(n) = f(0) + f(1) + f(2) + f(3) + ... + f(n-2) + f(n-1) = f(n-1) + f(n-1) 
    可以得出： 
    f(n) = 2*f(n-1) 
7) 得出最终结论,在n阶台阶，一次有1、2、...n阶的跳的方式时，总得跳法为： 
              | 1       ,(n=0 )  
f(n) =     | 1       ,(n=1 ) 
              | 2*f(n-1),(n>=2)
10.我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
比如n=3时，2*3的矩形块有3种覆盖方法：

思路：与第八题的跳台类似，斐波那契数列
public class Solution {
    public int RectCover(int target) {
        // 被覆盖的目标矩形的形状： 2*n
        // 每次新增加的一列，（1）如果竖着放对应的情况与 target为 n-1 时相同；
        // （2如果横着放，对应的情况与 target 为 n-2 时相同。
        if(target <=2){
            return target;
        }else{
            return RectCover(target-1) + RectCover(target-2);
        }
    }
}
*****11.输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
思路：如果一个整数不为0，那么这个整数至少有一位是1。如果我们把这个整数减1，那么原来处在整数最右边的1就会变为0，原来在1后面的所有的0都会变成1(如果最右边的1后面还有0的话)。其余所有位将不会受到影响。
举个例子：一个二进制数1100，从右边数起第三位是处于最右边的一个1。减去1后，第三位变成0，它后面的两位0变成了1，而前面的1保持不变，因此得到的结果是1011.我们发现减1的结果是把最右边的一个1开始的所有位都取反了。这个时候如果我们再把原来的整数和减去1之后的结果做与运算，从原来整数最右边一个1那一位开始所有位都会变成0。如1100&1011=1000.也就是说，把一个整数减去1，再和原整数做与运算，会把该整数最右边一个1变成0.那么一个整数的二进制有多少个1，就可以进行多少次这样的操作。
public class Solution {
    public int NumberOf1(int n) {
    int sum =0;  //该位用来数1的个数
        while(n!=0){
            sum++; //不等于0就自加一
            n=n&(n-1);
            
        }
        return sum;
    }
}}
12.给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
保证base和exponent不同时为0（递归求解）
public class Solution {
    public static double Power(double base, int exp) {
        boolean flag = false;  //先定义一个标志位flag
        if (exp < 0) {  //如果次方数小于0，exp就取负数
            flag = true;
            exp = -exp;
        }
        double ans = 1;  //定义一个ans =1
        while (exp > 0) {//如果次方数大于0且次方数等于1时，数值就等于base本身
            if ((exp & 1) == 1) {
                ans = ans * base;
            }            //exp >>= 1; 指数减小一半，然后把结果平方，再判断指数是不是奇数，是奇数还得再乘一个
            exp >>= 1;  //这里应该是次方数不为0和1的其他情况，base=base*base
            base *= base;
        }
        return flag ? 1 / ans : ans;  //这里返回的是什么
    }
}
*****13.输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。（这题还不懂）
public class Solution {
    public void reOrderArray(int [] array) {
        if(array==null || array.length ==0){//数组为null或长度为0就不返回
            return ;
        }
        int m = 0;
        for(int i=0;i<array.length;i++){  //遍历数组
            if((Math.abs(array[i]))%2!=0){//是奇数的话，将数字存入tmp，在标记一个j为当前i
                int tmp = array[i]; //以下不懂
                int j=i;
                while(j>m){//当j比m的数值要大的话，把前一位的数据移到后一位上
                    array[j] = array[j-1];
                    j--;
                }
                m=j+1;
                array[j] = tmp;
            }
        }
    }
}
*****14.输入一个链表，输出该链表中倒数第k个结点。（和15题类似）
/*
public class ListNode {
    int val;
    ListNode next = null;
    ListNode(int val) {
        this.val = val;
    }
}*/
public class Solution {
    public ListNode FindKthToTail(ListNode head,int k) {
        if(head == null || k ==0 ){ //首先判断链表和第k位是否为null，判断链表为空或长度为1的情况
            return null;
        }
        ListNode slow=head;  //定义两个链表slow和fast均为原链表head
        ListNode fast=head;
        for(int i=0;i<k;i++){//从链表第一位遍历到第k位，如果fast是null就返回null
            if(fast==null){
                return null;
            }
            fast=fast.next;//如果fast不为null就指向下一节点fast.next
        }
        while(fast!=null){//以下看不懂，fast不为null就一直往下走节点，直到找到第k个节点
            slow=slow.next;//记录slow的下一节点位置赋值给slow
            fast=fast.next;//记录fast的下一节点位置赋值给fast
        }
        return slow;  //fast为null就返回slow的值
    }
}
***15.输入一个链表，反转链表后，输出新链表的表头。（关于链表的题还需要多看）
/*
public class ListNode {
    int val;
    ListNode next = null;
    ListNode(int val) {
        this.val = val;
    }
}*/
public class Solution {
    public ListNode ReverseList(ListNode head) {
         // 判断链表为空或长度为1的情况
        if(head == null || head.next == null){
            return head;
        }
        ListNode pre = null; // 当前节点的前一个节点
        ListNode next = null; // 当前节点的下一个节点
        while( head != null){  //如果当前节点不为null
            next = head.next; // 记录当前节点的下一个节点位置；
            head.next = pre; // 让当前节点指向前一个节点位置，完成反转
            pre = head;      // pre 往右走
            head = next;     // 当前节点往右继续走，一共pre、head、next三个顺序节点，pre走到head的位置，head走到next的位置，pre和next集体右移一位，而next指向next的下一个节点，以此类推
        }
        return pre;
    }
}
**16.输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
public ListNode Merge(ListNode list1, ListNode list2) {
        if(list1==null)  //特例情况
            return list2;
        if(list2==null)
            return list1;
        ListNode res = null;  //合成后的链表res
        if(list1.val<list2.val){ //链表1中的第一个数的值val小于链表2中的val时，合成后的链表返回list1
            res = list1;
            res.next = Merge(list1.next, list2); //合成链表的下一位就是链表1的下一位和链表2的该位比较
        }else{  //大于等于时
            res = list2;
            res.next = Merge(list1, list2.next);
        }
        return res;
    }
思路：给两个链表分别添加一个“指针”，从第一个元素开始比较，小的输出，输出的链表再指向下一个元素，再进行比较，以此类推

***17.输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
public class Solution {
    //遍历大树
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        if(root1 == null || root2 == null){ //两棵树不能为null
            return false;
        }
        //如果找到与子树相同根的值，走判断方法
        if(root1.val == root2.val){
            if(judgeSubTree(root1,root2)){
                return true;
            }
        }//未找到与子根相同根的值
        //遍历左孩子，右孩子，这句不太理解，只要树A的左孩子与树B相同根或树A的右孩子与树B相同根即返回true
        return HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
    }
    //判断是否是子结构
    public boolean judgeSubTree(TreeNode root, TreeNode subtree) {
        //子结构已经循环完毕，代表全部匹配
        if(subtree == null){
            return true;
        }
        //大树已经循环完毕，并未成功匹配
        if(root == null){
            return false;
        }
        //相等后判断左右孩子
        if(root.val == subtree.val){
            return judgeSubTree(root.left, subtree.left) && judgeSubTree(root.right, subtree.right);
        }
        return false;
    }
}
18.操作给定的二叉树，将其变换为源二叉树的镜像。
二叉树的镜像定义：源二叉树                            	镜像二叉树
    	                 8                                          8
    	                /  \                                        /   \
    	               6   10                                     10   6
               	 /  \   /  \                                   /  \   /  \
             	5  7  9  11                               11   9  7  5
public class Solution {
    public void Mirror(TreeNode root) {
        if(root == null){  //特例判断二叉树是否为null
            return;
        }
        TreeNode temp = root.left;  //先用一个暂时变量temp存二叉树左侧数据
        root.left = root.right;    //将右侧数据赋值给左侧数据
        root.right = temp;         //将temp中存储的左侧数据赋值给右侧数据
        Mirror(root.left);       //镜像就行了
        Mirror(root.right);
    }
}
19.输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.（回字形打印）
思路：简单来说，就是不断地收缩矩阵的边界；定义四个变量代表范围，up、down、left、right
1.向右走存入整行的值，当存入后，该行再也不会被遍历，代表上边界的 up 加一，并判断是否和代表下边界的 down 交错 ；
2.向下走存入整列的值，当存入后，该列再也不会被遍历，代表右边界的 right 减一，判断是否和代表左边界的 left 交错； 
3.向左走存入整行的值，当存入后，该行再也不会被遍历，代表下边界的 down 减一，判断是否和代表上边界的 up 交错； 
4.向上走存入整列的值，当存入后，该列再也不会被遍历，代表左边界的 left 加一，判断是否和代表右边界的 right 交错；
import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> printMatrix(int [][] matrix) {
        ArrayList<Integer> list = new ArrayList<>();
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0){//特例判断
            return list;
        }
        int up = 0; //定义四个变量方位
        int down = matrix.length-1;
        int left = 0;
        int right = matrix[0].length-1;
        while(true){
            // 最上面一行，col为纵坐标，row为横坐标
            for(int col=left;col<=right;col++){ //从左遍历到右，数组中元素++
                list.add(matrix[up][col]); //列表中添加最上面一行的数组元素
            }
            // 向下逼近
            up++; 
            // 判断是否越界
            if(up > down){
                break;
            }
            // 最右边一行
            for(int row=up;row<=down;row++){ //最右边一列，从最上面一个up遍历到最下面一个down
                list.add(matrix[row][right]); //在列表中添加最右一列数组元素
            }
            // 向左逼近
            right--;
            // 判断是否越界
            if(left > right){
                break;
            }
            // 最下面一行
            for(int col=right;col>=left;col--){ //最下面一行，从最右遍历到最左--
                list.add(matrix[down][col]);
            }
            // 向上逼近
            down--;
            // 判断是否越界
            if(up > down){
                break;
            }
            // 最左边一行
            for(int row=down;row>=up;row--){ //最左一列，从最下遍历到最上--
                list.add(matrix[row][left]);
            }
            // 向右逼近
            left++;
            // 判断是否越界
            if(left > right){
                break;
            }
        }
        return list;
    }
}
***20.定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
注意：保证测试中不会当栈为空的时候，对栈调用pop()或者min()或者top()方法。
import java.util.Stack;
public class Solution {
    Stack<Integer> stackTotal = new Stack<Integer>(); //定义了两个栈，一个用来存所有元素，
Stack<Integer> stackLittle = new Stack<Integer>();//一个用来存加入新元素后的最小值

    public void push(int node) { //入栈操作
        stackTotal.push(node); //将元素依次压入栈内
        if(stackLittle.empty()){ //如果此时的最小值栈为空的话就将第一个元素压进去
            stackLittle.push(node);
        }else{                    //最小值栈不为空就判断下一个元素和最小值栈的栈顶元素大小
            if(node <= stackLittle.peek()){//如果下一个元素小于栈顶元素，就将该元素压入最小值栈
                stackLittle.push(node);
            }else{                           //如果下一元素大于栈顶元素，就再压入一次最小值栈的原栈顶
                stackLittle.push(stackLittle.peek());
            }
        }
    }
    public void pop() {  //弹出各自栈顶元素
        stackTotal.pop();
        stackLittle.pop();
    }
    public int top() {
        return stackTotal.peek();
    }
    public int min() {
        return stackLittle.peek();
    }
}
两个栈中的元素数量始终保持一致，当新的元素小于“stackLittle”栈顶元素时，“stackLittle”像栈顶push新来的元素，否则，“stackLittle”向栈顶加入原栈顶元素。执行“pop”方法时，两个栈同时弹出各自的栈顶元素。
21.输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
思路：新建一个栈，将数组A压入栈中，当栈顶元素等于数组B时，就将其出栈，当循环结束时，判断栈是否为空，若为空则返回true.
import java.util.ArrayList;
import java.util.Stack;
public class Solution {
    public boolean IsPopOrder(int [] pushA,int [] popA) {
        if (pushA.length == 0 || popA.length == 0 || popA.length != pushA.length) //特例排除
            return false;
        Stack<Integer> stack = new Stack<>(); //新建一个栈
        int j = 0;
        for (int i = 0; i < pushA.length; i++) {  //遍历数组A中元素，将其压入栈内
            stack.push(pushA[i]);
 
            while (!stack.isEmpty() && stack.peek() == popA[j]){//如果栈不为空且栈顶元素和A的一致就弹出
                stack.pop();
                j++;
            }
        }
        return stack.isEmpty();
    }
}
*****22.从上往下打印出二叉树的每个节点，同层节点从左至右打印。
思路：
在Java中Queue是和List、Map同等级别的接口，LinkedList中也实现了Queue接口，该接口中的主要函数有：
1.容量不够或队列为空时不会抛异常：offer（添加队尾元素）、peek（访问队头元素）、poll（访问队头元素并移除）
2.容量不够或队列为空时抛异常：add、element（访问队列元素）、remove（访问队头元素并移除）
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;
    public TreeNode(int val) {
        this.val = val;
    }
}
*/
public class Solution {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {   //定义从上到下的函数
        ArrayList<Integer> result = new ArrayList<Integer>(); //新建数组列表result//result用来保存输出的节点
        if(root == null)return result;                        //如果二叉树为空则返回结果
//queue用来保存当前遍历到了哪个节点，一次性把一个节点的左右子都入队
        Queue<TreeNode> queue = new LinkedList<TreeNode>(); //新建队列queue
        queue.offer(root);                  //向队列中添加二叉树元素
//只要队列中还有节点就说明还没遍历完，继续。
        //每次从队列出队，然后将这个节点左右子入队列
        while(!queue.isEmpty()){           //当队列内不为空时，
            TreeNode temp = queue.poll();  //将队列开头元素存入temp
            result.add(temp.val);          //在result中添加temp的值，如果temp左孩子不为空就添加左孩子
            if(temp.left != null)queue.offer(temp.left);  //有左孩子则入队
            if(temp.right != null)queue.offer(temp.right); //同理，temp右孩子不为空就向队列添加右孩子
        }
        return result;
    }
}
*****23.输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。（后序遍历是左孩子右孩子根节点）
public class Solution {
   public boolean VerifySquenceOfBST(int [] sequence) {  //BST为二叉树
        if(sequence == null || sequence.length == 0)
return false; //特例情况
           return  helpVerify(sequence, 0, sequence.length-1);
    }
     public boolean helpVerify(int [] sequence, int start, int root){
        if(start >= root)return true;
        int key = sequence[root]; //根节点
        int i;
        //找到左右子数的分界点
        for(i=start; i < root; i++)  //从开始遍历二叉树，如果顺序比根节点的位置要大就跳出，区分左右孩子
            if(sequence[i] > key)
                break;
        //在右子树中判断是否含有小于root的值，如果有返回false
        for(int j = i; j < root; j++)
            if(sequence[j] < key)  //以下都看不懂？？
                return false;
        return helpVerify(sequence, start, i-1) && helpVerify(sequence, i, root-1);
    }
}
*****24.输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
public class Solution {
private ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();  
    private ArrayList<Integer> list = new ArrayList<>();   //定义两个数组列表result和list
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        if(root == null)return result; 
        list.add(root.val);  //在队列中加入二叉树的值
        target -= root.val;
        if(target == 0 && root.left == null && root.right == null) //如果目标值为0且左右孩子均为空
//这里为什么用ArrayList，因为result.add(list)是把list这个对象的引用地址添加到result了，result中的元素就会共用list，而list是我们用来存放当前路径的地方，因此我们需要复制一份之后加入result数组中
            result.add(new ArrayList<Integer>(list));               //就将list添加到result中
//因为在每一次的递归中，我们使用的是相同的result引用，所以其实左右子树递归得到的结果我们不需要关心，
//可以简写为FindPath(root.left, target)；FindPath(root.right, target)；
//但是为了大家能够看清楚递归的真相，此处我还是把递归的形式给大家展现了出来。
        ArrayList<ArrayList<Integer>> result1 = FindPath(root.left, target);
        ArrayList<ArrayList<Integer>> result2 = FindPath(root.right, target);
        list.remove(list.size()-1);//因为当本次递归结束返回上一层的时候，我们已经遍历完这个节点的左右子树，也就是已经该树中可能存在的路径，再次返回上一层的时候要把这个节点挪除去，这样在遍历上一个节点的其他子树的时候遍历的结果才是对的
        return result;
    }
}
*******************25.输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）（这题方法还看不懂）
思路：
遍历链表，复制每个结点，如复制结点A得到A1，将结点A1插到结点A后面；
重新遍历链表，复制老结点的随机指针给新结点，如A1.random = A.random.next;
拆分链表，将链表拆分为原链表和复制后的链表

public class Solution {
    public RandomListNode Clone(RandomListNode pHead) {
        if(pHead == null) { //特例情况
            return null;
        }
        RandomListNode currentNode = pHead;
        //1、复制每个结点，如复制结点A得到A1，将结点A1插到结点A后面；
        while(currentNode != null){
            RandomListNode cloneNode = new RandomListNode(currentNode.label); //复制结点
            RandomListNode nextNode = currentNode.next; //将复制得到的点插入原结点后
            currentNode.next = cloneNode;  //也就是原结点的下一个是复制结点，
            cloneNode.next = nextNode;   //复制结点下一个是原来的下一个结点，
            currentNode = nextNode;//插入链表
        }
 
        currentNode = pHead;
        //2、重新遍历链表，复制老结点的随机指针给新结点，如A1.random = A.random.next;
        while(currentNode != null) {  //如果现在结点不为空就复制老结点的随机指针
//这句是如果老结点的随机指针为null就返回新结点为null，如果不为空就复制老结点的随机指针给新结点
            currentNode.next.random = currentNode.random==null?null:currentNode.random.next;
            currentNode = currentNode.next.next; //这句什么意思？？应该就是将现有节点的复制节点的下一位赋给现有节点，.next代表复制节点
        }
 
        //3、拆分链表，将链表拆分为原链表和复制后的链表
        currentNode = pHead;                      //老结点
        RandomListNode pCloneHead = pHead.next; //复制后的随机指针
        while(currentNode != null) {  //如果老结点不为null，则将链表拆分
            RandomListNode cloneNode = currentNode.next;
            currentNode.next = cloneNode.next;  //拆分后的原链表
            cloneNode.next = cloneNode.next==null?null:cloneNode.next.next; //拆分后复制的链表
            currentNode = currentNode.next;
        }
        return pCloneHead;  //返回复制的链表
    }
}
*****26.输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
思路：中序遍历二叉树，然后用一个ArrayList类保存遍历的结果，这样在ArratList中节点就按顺序保存了，然后再来修改指针
Import java.util.ArrayList;
public class Solution {
public TreeNode Convert(TreeNode pRootOfTree) {
        if(pRootOfTree == null){  //特例情况
            return null;
        }
        ArrayList<TreeNode> list = new ArrayList<>(); //新建一个数组列表list用来保存遍历结果
        Convert(pRootOfTree, list);
        return Convert(list);
    }
    //中序遍历，在list中按遍历顺序保存
    public void Convert(TreeNode pRootOfTree, ArrayList<TreeNode> list){
        if(pRootOfTree.left != null){  //二叉搜索树左孩子不为空，写入list
            Convert(pRootOfTree.left, list);
        }
        list.add(pRootOfTree); 
        if(pRootOfTree.right != null){ //二叉搜索树右孩子不为空，也写入list
            Convert(pRootOfTree.right, list);
        }
    }
    //遍历list，修改指针
    public TreeNode Convert(ArrayList<TreeNode> list){
        for(int i = 0; i < list.size() - 1; i++){ //遍历list
            list.get(i).right = list.get(i + 1); //修改list指针
            list.get(i + 1).left = list.get(i);
        }
        return list.get(0);
    }
}
*****27.输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
思路：
假设输入为a、b、c
那么其实排序的总数：
fun（a，b，c）=a（fun（b，c））+ a和b交换（fun（a，c））+a和c交换（fun（b，a））
fun（b，c） = b+fun（c）+b和c交换（fun（b））
fun（c）=1
所以用递归的方法就可以了，并且在这个递归的过程中，并没有做出一些浪费运行时间的事情，每一个递归都会产生新的结果，因此用递归来解决被称为动态规划的此题，是合理的。
另外题目中说明可能存在重复的字符，因此在进行交换的时候需要判断进行交换的字符是否相等，如果相等就没有必要交换了。
import java.util.ArrayList;
import java.util.Collections;
public class Solution {
    public ArrayList<String> PermutationHelp(StringBuilder str){
         ArrayList<String> result = new  ArrayList<String>(); //新建数组列表
        if(str.length() == 1)result.add(str.toString());   //特例判断
        else{
            for(int i = 0; i < str.length(); i++){ //遍历str
                if(i== 0  || str.charAt(i) != str.charAt(0)){  //如果不是第一位的话
                    char temp = str.charAt(i);
                    str.setCharAt(i, str.charAt(0)); //则在第一位0处设置一个字符
                    str.setCharAt(0, temp);          //在str中设置从0到该位的字符
                    ArrayList<String> newResult = PermutationHelp(new StringBuilder(str.substring(1)));               
                    for(int j =0; j < newResult.size(); j++)
                        result.add(str.substring(0,1)+newResult.get(j)); //获取[0,1)的字符串添加result
                    //用完还是要放回去的
                    temp = str.charAt(0);
                    str.setCharAt(0, str.charAt(i));
                    str.setCharAt(i, temp);
                }
            }
            //需要在做一个排序操作
        }
         Collections.sort(result);  //新增部分
        return result;
    }
    public ArrayList<String> Permutation(String str) {
        StringBuilder strBuilder = new StringBuilder(str);
        ArrayList<String> result = PermutationHelp(strBuilder);
        return result;
    }
}
28.数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
思路：
用一般的排序也可以完成这道题目，但是如果那样完成的话就可能太简单了。
用preValue记录上一次访问的值，count表明当前值出现的次数，如果下一个值和当前值相同那么count++；如果不同count--，减到0的时候就要更换新的preValue值了，因为如果存在超过数组长度一半的值，那么最后preValue一定会是该值。
public class Solution {
    public int MoreThanHalfNum_Solution(int [] array) {
        if(array == null || array.length == 0)return 0; //数组为空的特例情况
        int preValue = array[0];  //用来记录上一次的记录
        int count = 1;            //preValue出现的次数（相减之后）
        for(int i = 1; i < array.length; i++){  //遍历数组
            if(array[i] == preValue)
                count++;
            else{
                count--;
                if(count == 0){  //如果count减到0就更换新的preValue值
                    preValue = array[i];
                    count = 1;
                }
            }
        }
        int num = 0;      //需要判断是否真的是大于一半数
        for(int i=0; i < array.length; i++)
            if(array[i] == preValue)
                num++;
        return (num > array.length/2)?preValue:0; //如果num数大于数组长度的一般就输出，否而输出0
    }
}
29.输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
思路：
维持一个K长度的最小值集合，然后利用插入排序的思想进行对前K个元素的不断更新。但是非常让人气愤的是居然if(k<= 0 || k > input.length)return result的判断占据了用例的发部分。
import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        if(k<= 0 || k > input.length)return result; //特例情况
        //初次排序，完成k个元素的排序
        for(int i = 1; i< k; i++){
            int j = i-1;
            int unFindElement = input[i];
            while(j >= 0 && input[j] > unFindElement){
                input[j+1] = input[j];
                j--;
            }
            input[j+1] = unFindElement;
        }
        //遍历后面的元素 进行k个元素的更新和替换
        for(int i = k; i < input.length; i++){
            if(input[i] < input[k-1]){
                int newK = input[i]; //如果新的元素比原元素小酒替换
                int j = k-1;
                while(j >= 0 && input[j] > newK){
                    input[j+1] = input[j];
                    j--;
                }
                input[j+1] = newK;
            }
        }
        //把前k个元素返回
        for(int i=0; i < k; i++)
            result.add(input[i]);
        return result;
    }
}
*****30.HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)
思路：这题目应该是最基础的动态规划的题目：最大子数组的和一定是由当前元素和之前最大连续子数组的和叠加在一起形成的，因此需要遍历n个元素，看看当前元素和其之前的最大连续子数组的和能够创造新的最大值。
public class Solution {
    public int FindGreatestSumOfSubArray(int[] array) {
        int len = array.length; //定义数组长度len
        int[] dp = new int[len]; 
        int max = array[0]; //先设定最大值是数组中的第一个元素
        dp[0] = array[0];  //dp[i]的含义是以第i个元素为右边界的连续子序列所能取到的最大值
        for(int i=1; i < len; i++){  //遍历数组
        //最大值newMax等于以第i-1个元素为右边界的连续子序列所能取到的最大值+数组中第i个元素
            int newMax = dp[i-1] + array[i];//这里的判断条件有些看不懂
            if(newMax > array[i])  //如果最大值比第i个元素要大的话，证明dp[i-1]里面没有负数
                dp[i] = newMax; //则dp[i]就是最大值newMax
            else                //如果最大值比第i个元素要小的话，代表里面有负数
                dp[i] = array[i]; //则dp[i]就由数组第i个元素赋值
            if(dp[i] > max)  //如果dp[i]比最大值大就替换最大值
                max = dp[i];
        }
        return max;
    }
}
*****31.求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。
思路：
其实这道题目需要比较深刻的数学归纳计算法进行计算，才能够用几行代码就可以解决。
个位数上的1的个数：(n/10)1 + if(n%10)<1?0:((n%10-1)+1)  
//每隔10次会出现1次1，如果数值n取余10的值小于1那么个位就没有1，如果取余大于1就将取余值减一后加1
十位数上的1的个数：(n/100)10 + if(n%100)<10?0:((n%100-10)+1) //每隔100次会出现10次1
百位数上1的个数：(n/1000)100 + if(n%1000)<100?0:((n%1000-100)+1)  
总计i：*(n/(10i))i + if(n%(10i))< i ?0:((n%(10*i)-i)+1)
i是小于n的最高位数目：i=pow(10, log10(n))
public class Solution {
    public int NumberOf1Between1AndN_Solution(int n) {
        if(n <= 0)return 0;  //特例情况
        int count = 0;  //存储有1 的数的个数
        for(int i=1; i <= n; i*=10){ //遍历n个数
            //计算在第i位上总共有多少个1
            count = count + (n/(10*i))*i; //第一位就n/10*1，第二位是n/100*2.
            //不足i的部分有可能存在1
            int mod = n%(10*i);
            //如果超出2*i -1，则多出的1的个数是固定的
            if(mod > 2*i -1)count+=i;
            else{
                //只有大于i的时候才能会存在1
                if(mod >= i)
                    count += (mod -i)+1;
            }
        }
        return count;
    }
}
32.输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
思路：如何将各个元素从小到大进行排序，排序之后再把他们串联起来就可以了，简直是非常的机智，非常的完美呀。
比较关键的一句话 所以在这里自定义一个比较大小的函数，比较两个字符串s1, s2大小的时候，先将它们拼接起来，比较s1+s2,和s2+s1那个大，如果s1+s2大，那说明s2应该放前面，所以按这个规则，s2就应该排在s1前面。
import java.util.ArrayList;
public class Solution {
    public String PrintMinNumber(int [] numbers) {
        if(numbers == null || numbers.length == 0)return "";  //特例情况
        for(int i=0; i < numbers.length; i++){  //遍历数组中相邻元素
            for(int j = i+1; j < numbers.length; j++){
                int sum1 = Integer.valueOf(numbers[i]+""+numbers[j]);  //sum1为顺序相邻两位相加的数值，
                int sum2 = Integer.valueOf(numbers[j]+""+numbers[i]); //sum2为反序相邻两位相加的数值
                if(sum1 > sum2){   //如果第一个元素大就互换两者位置，让sum2在前
                    int temp = numbers[j];
                    numbers[j] = numbers[i];
                    numbers[i] = temp;
                }
            }
        }
        String str = new String("");      //定义一个字符串str
        for(int i=0; i < numbers.length; i++)  //遍历排好序的数组
            str = str + numbers[i];  //依次相加得到最小数字
        return str;
    }
}
*****33.把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
思路：
这道题目自己是有思路的，丑数能够分解成2^x3^y5^z,
所以只需要把得到的丑数不断地乘以2、3、5之后并放入他们应该放置的位置即可，
而此题的难点就在于如何有序的放在合适的位置。
1乘以 （2、3、5）=2、3、5；2乘以（2、3、5）=4、6、10；3乘以（2、3、5）=6,9,15；5乘以（2、3、5）=10、15、25；
从这里我们可以看到如果不加策略地添加丑数是会有重复并且无序，
而在2x，3y，5z中，如果x=y=z那么最小丑数一定是乘以2的，但关键是有可能存在x》y》z的情况，所以我们要维持三个指针来记录当前乘以2、乘以3、乘以5的最小值，然后当其被选为新的最小值后，要把相应的指针+1；因为这个指针会逐渐遍历整个数组，因此最终数组中的每一个值都会被乘以2、乘以3、乘以5，也就是实现了我们最开始的想法，只不过不是同时成乘以2、3、5，而是在需要的时候乘以2、3、5.
public class Solution {
    public int GetUglyNumber_Solution(int index) {
        if(index <= 0)return 0;  //特例情况
        int p2=0,p3=0,p5=0;      //初始化三个指向三个潜在成为最小丑数的位置
        int[] result = new int[index];  
        result[0] = 1;         //1是当作第一个丑数
        for(int i=1; i < index; i++){  //遍历index
             //找各个三个指向的最小值
            result[i] = Math.min(result[p2]*2, Math.min(result[p3]*3, result[p5]*5)); 
             //这里找到一个最小值该指针指向下一位
            if(result[i] == result[p2]*2)p2++;//为了防止重复需要三个if都能够走到 
            if(result[i] == result[p3]*3)p3++;//为了防止重复需要三个if都能够走到
            if(result[i] == result[p5]*5)p5++;//为了防止重复需要三个if都能够走到
        }
        return result[index-1];
    }
}
***34.在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
思路：按照hash的思想来做的，先统计出现的次数，然后在返回相应的index
public class Solution {
    public int FirstNotRepeatingChar(String str) {
        if(str==null || str.length() == 0)return -1; //特例情况
        int[] count = new int[256];  //定义一个256大小的count
        //用一个类似hash的东西来存储字符出现的次数，很方便
        for(int i=0; i < str.length();i++)
            count[str.charAt(i)]++;
        //其实这个第二步应该也是ka我的地方，没有在第一时间想到只要在遍历一遍数组并访问hash记录就可以了
        for(int i=0; i < str.length();i++)
            if(count[str.charAt(i)]==1)
                return i;
        return -1;
    }
}
*****35.在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

思路：
所以不得已只能用大家提供的归并排序的思路来做这道题目以实现o（nlogn）的复杂度。
在归并排序的过程中 后一个数组的数如小于前一个数组的数，则一定能够构成逆序对且逆序对的数目可计算，因为待归并的两个数组提前已经归并排序过，所以不会出现像前面那样少统计或者多统计的情况出现。
思路：[A，B]中的逆序对=[A]的逆序对+[B]中的逆序对+将A，B混排在一起的逆序对
而将A，B混排在一起的逆序对求解看下面：
public class Solution {
    private int cnt;
    private void MergeSort(int[] array, int start, int end){
        if(start>=end)return;  //特例情况
        int mid = (start+end)/2;
        MergeSort(array, start, mid);  //MergeSort是归并排序
        MergeSort(array, mid+1, end);
        MergeOne(array, start, mid, end);
    }
    private void MergeOne(int[] array, int start, int mid, int end){
        int[] temp = new int[end-start+1];
        int k=0,i=start,j=mid+1;
        while(i<=mid && j<= end){   //以下排序看不太懂？？
//如果前面的元素小于后面的不能构成逆序对
            if(array[i] <= array[j])
                temp[k++] = array[i++];
            else{
//如果前面的元素大于后面的，那么在前面元素之后的元素都能和后面的元素构成逆序对
                temp[k++] = array[j++];
                cnt = (cnt + (mid-i+1))%1000000007;
            }
        }//以下并归排序一次合并排序的最后步骤，为防止最后一个数组中还剩有多个元素需要存放到新的已排序的数组中
        while(i<= mid)  
            temp[k++] = array[i++];
        while(j<=end)
            temp[k++] = array[j++];
        for(int l=0; l<k; l++){ //注意这里的l
            array[start+l] = temp[l];
        }
    }
    public int InversePairs(int [] array) {
        MergeSort(array, 0, array.length-1);
        return cnt;
    }
}
*****36.输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）
思路：遍历两遍这两个链表，如果有重复的节点，那么一定能够使遍历的指针相等。
看下面的链表例子：
0-1-2-3-4-5-null
a-b-4-5-null
代码的ifelse语句，对于某个指针p1来说，其实就是让它跑了连接好的的链表，长度就变成一样了。
如果有公共结点，那么指针一起走到末尾的部分，也就一定会重叠。看看下面指针的路径吧。
p1： 0-1-2-3-4-5-null(此时遇到ifelse)-a-b-4-5-null
p2: a-b-4-5-null(此时遇到ifelse)0-1-2-3-4-5-null
因此，两个指针所要遍历的链表就长度一样了！
如果两个链表存在公共结点，那么p1就是该结点，如果不存在那么p1将会是null。
public class Solution {
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if(pHead1 == null || pHead2 == null)return null; //特例情况
        ListNode p1 = pHead1; //定义p1指针，比较的是结点的引用地址
        ListNode p2 = pHead2;
        while(p1!=p2){  //如果两个指针不相等就各自指向下一个
            p1 = p1.next;
            p2 = p2.next;
            if(p1 != p2){ //两个指针不相等的时候，，p1为null时p1指向链表2，同理p2
                if(p1 == null)p1 = pHead2;  //代表第一个链表已经访问结束，还没找到公共节点，需要接着遍历
                if(p2 == null)p2 = pHead1;
            }
        }
        return p1;
    }
}
37.统计一个数字在排序数组中出现的次数。
思路：用java中的排序即可
import java.util.Arrays;
public class Solution {
    public int GetNumberOfK(int [] array , int k) {
        int index = Arrays.binarySearch(array, k);  //index是数字k在数组array中的位置
        if(index<0)return 0;  //特例情况
        int cnt = 1;  //定义的是数字出现的次数
        for(int i=index+1; i < array.length && array[i]==k;i++)  //遍历数组array找到数组中数字为k的值就自加一
            cnt++;
        for(int i=index-1; i >= 0 && array[i]==k;i--) //这里的index-1和上面的index+1分别是从哪个位置开始的？？？（这里就是分两段进行遍历，上面是找数字k位置之后的数字中，这里是找数字k位置之前的数字中）
            cnt++;
        return cnt;
    }
}
38.输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
思路：用递归算法即可
public class Solution {
    public int TreeDepth(TreeNode root) {
        if(root == null)return 0; //特例情况
        int leftDepth = TreeDepth(root.left); //定义左子树的深度leftDepth
        int rightDepth = TreeDepth(root.right); //右子树的深度rightDepth
        int result = 1 + ((leftDepth > rightDepth)?leftDepth:rightDepth); //result代表深度=1（1是最上面的树头）+左右子树哪个大就走哪条路
        return result;
    }
}
***39.输入一棵二叉树，判断该二叉树是否是平衡二叉树。
思路：平衡二叉树的左右子树也是平衡二叉树，那么所谓平衡就是左右子树的高度差不超过1.
public class Solution {
    public int depth(TreeNode root){
        if(root == null)return 0;//特例情况
        int left = depth(root.left); //定义左右子树深度
        if(left == -1)return -1; //如果发现子树不平衡之后就没有必要进行下面的高度的求解了
        int right = depth(root.right);
        if(right == -1)return -1;//如果发现子树不平衡之后就没有必要进行下面的高度的求解了
        if(left - right <(-1) || left - right > 1)//判断如果左右子树深度差大于1的话也不用考虑了
            return -1;
        else
            return 1+(left > right?left:right); //如果左右子树深度差为1及以内，谁深度大就返回谁
    }
    public boolean IsBalanced_Solution(TreeNode root) {
        return depth(root) != -1;
    }
}
*****40.一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
思路：
首先：位运算中异或的性质：两个相同数字异或=0，一个数和0异或还是它本身。
当只有一个数出现一次时，我们把数组中所有的数，依次异或运算，最后剩下的就是落单的数，因为成对儿出现的都抵消了。
依照这个思路，我们来看两个数（我们假设是AB）出现一次的数组。我们首先还是先异或，剩下的数字肯定是A、B异或的结果，这个结果的二进制中的1，表现的是A和B的不同的位。我们就取第一个1所在的位数，假设是第3位，接着把原数组分成两组，分组标准是第3位是否为1。如此，相同的数肯定在一个组，因为相同数字所有位都相同，而不同的数，肯定不在一组。然后把这两个组按照最开始的思路，依次异或，剩余的两个结果就是这两个只出现一次的数字。
public class Solution{
public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
        int xor1 = 0;  //定义的是一个数xor1为0
        for(int i=0; i < array.length; i++)  //遍历数组
            xor1 = xor1^array[i]; //一个数和0异或还是该数本身，也就是把该数赋值给xor1
        //在xor1中找到第一个不同的位对数据进行分类，分类为两个队列对数据进行异或求和找到我们想要的结果
        int index = 1;
        while((index & xor1)==0) //index与xor1与运算代表的是什么？？？
            index = index <<1;//因为可能有多个位为1所以需要求一下位置
        int result1 = 0;
        int result2 = 0;
        for(int i=0; i < array.length; i++){  //异或之后基本看不懂？？？
            if((index & array[i]) == 0)
                result1 = result1^array[i];
            else
                result2 = result2^array[i];
        }
        num1[0] = result1;
        num2[0] = result2;
}
}
***41.小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!
输出描述：
输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
思路：穷举的方式
import java.util.ArrayList;
public class Solution {
    public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) { 
        ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>(); //定义数组列表result
        for(int i=1; i < sum; i++){ //遍历数sum中所有数
            int temp = 0;
            int j = i;
            while(temp < sum){ //如果temp小于总数sum的话就连续相加
                temp += j;
                j++;
            }
            if(temp == sum){//如果找到了那么就要把数据添加到结果数据中。
                ArrayList<Integer> newArray = new  ArrayList<Integer>();
                for(int k=i;k< j;k++)
                    newArray.add(k);
                result.add(newArray);
            }
        }
        return result;
    }
}
*****42.输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
输出描述：
对应每个测试案例，输出两个数，小的先输出
import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
        ArrayList<Integer> result=new ArrayList<Integer>(); //新建数组列表result
        //边界条件
        if(array==null||array.length<=1){//特例情况
            return result;
        }
        int smallIndex=0;  //定义小边界
        int bigIndex=array.length-1; //定义大边界
        while(smallIndex<bigIndex){ 
            //如果相等就放进去
             if((array[smallIndex]+array[bigIndex])==sum){
                result.add(array[smallIndex]);
                result.add(array[bigIndex]);
                 //最外层的乘积最小，别被题目误导
                 break;
        }else if((array[smallIndex]+array[bigIndex])<sum){
                 smallIndex++;
             }else{
                 bigIndex--;
             }
        }
        return result;
}
}
43.汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！
public class Solution {
    public String LeftRotateString(String str, int n) {
        if (str == null || n > str.length()) { //两个特例情况
            return str;
        }
        return str.substring(n) + str.substring(0, n); //返回循环后结果
    }
}
*****44.牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
思路：String没有reverse（）用法，StringBuffer.reverse（）返回的是String
import java.lang.StringBuffer;
public class Solution {
    public String ReverseSentence(String str) {
        if(str.length()<=0){ //特例情况
            return "";
        }
        //反转整个句子
        StringBuffer st1=new StringBuffer(str); //反转后的句子存入str1
        st1.reverse();
        //存放结果
         StringBuffer result=new StringBuffer();
         int j=0;
        //标记空格数
        int blankNum=0;
        for(int i=0;i<st1.length();i++){
            //1、当有空格，且没有到达最后一个单词时
            if(st1.charAt(i)==' '&&(i!=st1.length()-1)){
                blankNum++;  //空格数加1
                StringBuffer st2=new StringBuffer(st1.substring(j,i));
                result.append(st2.reverse().toString()).append(" ");
                j=i+1;
            }
           //2、当有空格，且到达最后一个单词时
            if(blankNum!=0&&i==(st1.length()-1)){
                 StringBuffer st3=new StringBuffer(st1.substring(j,i+1));
                result.append(st3.reverse());
            }
        }
        //空格数为0时，直接返回原字符串
        if(blankNum==0){
            return str;
        }
        return result.toString();
    }
}
*****45.LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
思路：可以这么理解，简单来说就是要是5个数字，最大和最小差值在5以内，并且没有重复数值。用一个set来填充数据，0不要放进去。set的大小加上0的个数必须为5个。此外set中数值差值在5以内。代码如下：
import java.util.TreeSet;
public class Solution {
    public boolean isContinuous(int [] n) {
        if (n.length < 5 || n.length > 5) { //首先第一个要求：是五个数，比5大或小都返回false，特例判断
            return false;
        }
        int num = 0;
        TreeSet<Integer> set = new TreeSet<> (); 
        for (int i=0; i<n.length;i++) {
            if (n[i]==0) { //如果抽到大/小王
                num ++;
            } else {
                set.add(n[i]);
            }
        }
        if ((num + set.size()) != 5) { //要求：set的大小加上0的个数不为5
            return false;
        }
        if ((set.last() - set.first()) < 5) {  //另一个要求：最大与最小值差值为5以内
            return true;
        }
        return false;
    }
}
*****46.每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
思路：java中直接使用一个list来模拟，并使用一个索引cur类指向删除的位置，当cur的值为list的size，就让cur到头位置。
import java.util.*;
public class Solution {
    public int LastRemaining_Solution(int n, int m) {
        if(n<1||m<1){ //特例情况
            return -1;
        }
        List<Integer> list = new ArrayList<>();
        //构建list
        for(int i = 0;i<n;i++){ //遍历数组，将数组中的数添加到list中
            list.add(i);
        }
        int cur = -1;      //先指定索引cur为-1
        while(list.size()>1){
            for(int i = 0;i<m;i++){ //从第0个开始，cur++，到索引cur的大小和list的size一致就将索引指0
                cur++;
                if(cur == list.size()){
                    cur = 0;
                }
            }
            list.remove(cur);
            cur--;//cur--的原因，因为新的list中cur指向了下一个元素，为了保证移动m个准确性，所以cur向前移动一位。
        }
        return list.get(0);
    }
}
*****47.求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
思路：
1.在编写代码时，首先需要取得乘数m的某一个进制位:
假设有变量 bitMask = 1，那么要取得 m 的第 k 位（从低位开始）二进制位的表达式就是： m & (bitMask << k);
当然还有其他写法，这里使用的是：(m >> k) & 1，即将该位位移到最低位，然后和1相与，屏蔽掉高位。 
2.其次需要根据该进制位的值对结果进行累加，如果值为0，则加0，如果值为1，则加上 (n << k):
因为题目不允许使用条件判断，所以这里还是通过位与运算来实现：num & 0x00000000 == 0，num & 0xFFFFFFFF == num
然后使用映射的思想，构建一个数组 mask，把0、1分别映射为0x00000000、0xFFFFFFFF
所以累加的表达式就为：result += (n << k) & mask[(m >> k) & 1];
一开始并没有想到可以使用“短路求值原理”来做为递归的结束条件，这时移位操作就可以写在参数里了，而不用写在表达式里，详细代码如下：
复杂度分析：虽然用到了递归，但是递归的执行次数最多为一个数二进制形式的长度，显然，整数n的二进制长度为log(n)
public class Solution {
    int[] mask = {0x00000000, 0xFFFFFFFF}; //构建一个数组 mask，把0、1分别映射为0x00000000、0xFFFFFFFF
    public int Sum_Solution(int n) {
        return production(n+1, n) >> 1;
    }
    int production(int m, int n) {
        int result = 0;  //用来存最后结果
        boolean isStop = (m != 0) &&                                            //结果等于判断条件不懂？？？
               (result = (n & mask[m & 1]) + production(m >> 1, n << 1)) != 0; //是否停止判断
        return result;
    }
}
  *****48.写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
思路：使用位运算实现加法。
1.一位加法
在位运算中，我们用“<<”表示向左移动一位，也就是“进位”。那么我们就可以得到如下的表达式:
( x & y ) << 1 
拥有了两个基本表达式：
执行加法 x ^ y 
进位操作 ( x & y ) << 1
2.二位加法
正确的加法计算：11+01 = 100 * 
 使用位运算实现二位加法：
按位加法： res1 = 11 ^ 01 = 10 
与运算进位： res2 = (11 & 01) << 1 = ( 01 ) << 1 = 010 
res1 ^ res2 = 10 ^ 010 = 00 
(10 & 10) << 1 = 100
3.三位加法
1）101 ^ 111 = 0010 （没有处理进位的加法）
(101 & 111) << 1 = 101 << 1 = 1010 （此处得到哪一位需要加上进位，为1的地方表示有进位需要加上）
2）0010 ^ 1010 = 1000 （没有处理进位的加法 + 进位 = 没有处理进位的加法）
  (0010 & 1010) << 1 = 0010 << 1 = 00100 （查看是否有新的进位需要处理）
3）1000 ^ 00100 （没有处理进位的加法 + 进位 = 没有处理进位的加法）
   (1000 & 00100) << 1 = 00000 << 1 = 000000 (进位为0，所以没有要处理的进位了)
public class Solution {
    public int Add(int num1,int num2) {
        int result = 0;
        int carry = 0;
        do{
            result = num1 ^ num2;       //不带进位的加法  异或门
            carry = (num1 & num2) << 1; //进位     与门
            num1 = result;  
            num2 = carry;   
        }while(carry != 0); // 进位不为0则继续执行加法处理进位
        return result;
    }
}
*****49.将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0

解法1：捕捉异常（有些钻空子嫌疑，但是很好用。。。。在题解中看到的）
public int StrToInt(String str) {
       Integer res=0;
        try {
             res = new Integer(str);
        } catch (NumberFormatException e) {
 
        } finally {
            return res;
        }
}
解法2：【最优解】
public class Solution{
public int StrToInt(String str) {
        //最优解
       if(str == null || "".equals(str.trim()))return 0; //特例判断
       str = str.trim();                               //字符串去空格
       char[] arr = str.toCharArray();
       int i = 0;
       int flag = 1;
       int res = 0;
       if(arr[i] == '-'){  //这里的判断条件看不懂？？这里应该是如果arr[i]是负数的话就让flag为-1
           flag = -1;
       }
       if( arr[i] == '+' || arr[i] == '-'){
           i++;
       }
       while(i<arr.length ){
           //是数字
           if(isNum(arr[i])){
               int cur = arr[i] - '0';
               if(flag == 1 && (res > Integer.MAX_VALUE/10 || res == Integer.MAX_VALUE/10 && cur >7)){
                   return 0;
               }
               if(flag == -1 && (res > Integer.MAX_VALUE/10 || res == Integer.MAX_VALUE/10 && cur >8)){
                   return 0;
           }
               res = res*10 +cur;
               i++;
           }else{
               //不是数字
               return 0;
           }
       }
       return res*flag;
   }
   public static boolean isNum(char c){
       return c>='0'&& c<='9';
   }
}
*****50.在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
思路：
解法一：排序。将输入数组排序，再判断相邻位置是否存在相同数字，如果存在，对 duplication 赋值返回，否则继续比较。
import java.util.*;
public class Solution {
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        if(numbers == null || length == 0){ //特例判断
            return false;
        }
        Arrays.sort(numbers);  //将数字输入数组
        for(int i=0;i<length-1;i++){ //遍历后判断相邻位置是否存在相同数字
            if(numbers[i] == numbers[i+1]){
                duplication[0] = numbers[i]; //存在就对duplication赋值返回
                return true;
            }
        }
        return false;
    }
}
时间复杂度：O(nlogn)
空间复杂度：O(1)
解法二：哈希表。利用 HashSet 解决，从头到尾扫描数组，每次扫描到一个数，判断当前数是否存在 HashSet 中，如果存在，则重复，对 duplication 赋值返回，否则将该数加入到 HashSet 中
import java.util.*;
public class Solution {
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        Set<Integer> set = new HashSet<>();
        for(int i =0 ;i<length;i++){ //从头到尾扫描数组，如果set中存在当前数
            if(set.contains(numbers[i])){
                duplication[0] = numbers[i]; //对duplication赋值返回
                return true;
            }else{
                set.add(numbers[i]);
            }
        }
        return false;
    }
}
时间复杂度：O(n)
空间复杂度：O(n)
*****51.给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。（注意：规定B[0] = A[1] * A[2] * ... * A[n-1]，B[n-1] = A[0] * A[1] * ... * A[n-2];）
思路：如下图所示先根据offer上的算法，先计算下三角的乘积，再计算上三角的乘积并且拼接

import java.util.ArrayList;
public class Solution {
    public int[] multiply(int[] A) {
        int length=A.length;
        int[] B=new int[length];
        //边界
        if(A==null||A.length<=1){ //特例情况
            return null;
        }
        //计算下三角
        //初始化第一行
        B[0]=1;
        for(int i=1;i<length;i++){
            B[i]=B[i-1]*A[i-1];
    }
        //计算上三角
        //初始化最后一行
        int temp=1;
        for(int i=length-1;i>=0;i--){
            B[i]=temp*B[i];
            temp=A[i]*temp;
        }
        return B;
}
}
*****52.请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配（判断条件很多，易弄混）
思路：
1.作对题目首先要读清题意：在本题中，匹配是指字符串的所有字符匹配整个模式。
2.模式串中可能存在'.*'，它是贪婪匹配，在使整个表达式能得到匹配的前提下匹配尽可能多的字符。例如字符串"abcdeded"与模式"a.*d"匹配。
3.按下一个字符是否是'*'分情况讨论，这个不难，但是要考虑全面有点难度。
4.c/c++一个函数递归就搞定，而java写要两个函数，因为你不能向c/c++一样直接用str+1,或用str+1传参。
public class Solution {
 public boolean matchStr(char[] str, int i, char[] pattern, int j) {
     // 边界
     if (i == str.length && j == pattern.length) { // 字符串和模式串都为空；一、二个都是空字符串，两者相等
         return true;
     } else if (j == pattern.length) { // 模式串为空，只有第二个为空字符串，两者应该不等
         return false;
     }
     boolean flag = false;
     boolean next = (j + 1 < pattern.length && pattern[j + 1] == '*'); // 模式串下一个字符是'*'
     if (next) {
         if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) { // 要保证i<str.length，否则越界
             return matchStr(str, i, pattern, j + 2) || matchStr(str, i + 1, pattern, j);//这句意思？？
         } else {
             return matchStr(str, i, pattern, j + 2);
         }
     } else {
         if (i < str.length && (pattern[j] == '.' || str[i] == pattern[j])) {
             return matchStr(str, i + 1, pattern, j + 1);
         } else {
             return false;
         }
     }
 }
 public boolean match(char[] str, char[] pattern) {
     return matchStr(str, 0, pattern, 0);
 }
}
*****53.请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
思路：正则表达式
^ 和 美元符号$框定正则表达式，它指引这个正则表达式对文本中的所有字符都进行匹配。如果省略这些标识，那么只要一个字符串中包含一个数字这个正则表达式就会进行匹配。如果仅包含 ^ ，它将匹配以一个数字开头的字符串。如果仅包含$ ，则匹配以一个数字结尾的字符串。
[-+]?
正负号后面的 ? 后缀表示这个负号是可选的,表示有0到1个负号或者正号
	\\d*
\d的含义和[0-9]一样。它匹配一个数字。后缀 * 指引它可匹配零个或者多个数字。
	(?:\\.\\d*)?
(?: …)?表示一个可选的非捕获型分组。* 指引这个分组会匹配后面跟随的0个或者多个数字的小数点。
	(?:[eE][+\\-]?\d+)?
这是另外一个可选的非捕获型分组。它会匹配一个e(或E)、一个可选的正负号以及一个或多个数字。
import java.util.regex.Pattern;
public class Solution {
public static boolean isNumeric(char[] str) {
//以^开头，$结尾的正则表达式；[-+]?表示可选0到1个负号或正号；\\d*匹配的是零个或多个数字；(?:\\.\\d*)?表示分组后面会跟0个或多个数字的小数点；(?:[eE][+\\-]?\\d+)?代表匹配一个e、一个可选的正负号以及一个或多个数字
        String pattern = "^[-+]?\\d*(?:\\.\\d*)?(?:[eE][+\\-]?\\d+)?$";
        String s = new String(str);
        return Pattern.matches(pattern,s);
    }
}
*****54.请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
输出描述：如果当前字符流没有存在出现一次的字符，返回#字符。
思路：
字符出现次数的判断（不重复字符）：
这个做法大致相同，利用 Hash 思想采用128大小的计数数组进行计数也好，或者是使用 Map 键值对映射也好，都差不多，使用数组会更简单。
字符出现顺序的判断（第一个字符）：
这里就是改进的关键之处了，容易发现，字符流中不重复的字符可能同时存在多个，我们只要把这些 “不重复字符” 保存起来就可以，而无需保存那些重复出现的字符，而为了维护字符出现的顺序，我们使用队列（先进先出）这一结构，先出现的不重复字符先输出：
入队：获取字符流中的一个字符时，当我们判断它是不重复时，将它加入队列； 
输出/出队：注意，因为队列中存储的 “不重复字符” 在一系列的流读取操作后，随时有可能改变状态（变重复），所以，队列中的字符不能直接输出，要先进行一次重复判断，如果发现队头字符已经重复了，就将它移出队列并判断新的队头，否则，输出队头的值； 
复杂度计算：
从上面的描述来看，好像存在一个循环，队列的长度好像无边无际，就给人一种O(n)的感觉，其实，并不是，有如下结论：
1.通过分析可以发现，循环（出队）的最大次数其实就是队列的长度，而队列的长度最大为128； 
2.并且随着时间的推移，队列长度 总体 先增大，后减小，正常条件下，最终队列会为空（因为随着字符流的增大，重复的字符会越来越多，队列就会不断地移除元素而越来越短）； 
3.更愉快的是，如果队列长度不减小，则循环就只执行一次，返回速度快，如果队列长度减小了，那么，循环次数上限也就减小了； 
所以时间、空间复杂度是一致的，都是常数级，可是这是为什么呢，分析如下：
1.字符的重复判断，因为使用的是直接 Hash，而且功能是计数，没有冲突，所以是O(1)； 
2.只有不重复的字符才入队列，但是不重复的字符有几个呢？ASCII字符最多也就128个，那么同一字符会不会多次入队呢？ 不会的，见3； 
3.只有队头元素变得重复了才执行循环，所以执行循环就意味着队列长度要变小。要注意，根据题意，字符的出现次数只增不减！！！所以，出队的字符不会再入队，队列长度总体上只会越来越小（或者上升到峰值就不再上升了，128种字符用尽）。
import java.util.Queue;
import java.util.LinkedList;
import java.lang.Character;
public class Solution {
    int[] charCnt = new int[128];                        //先定义一个长度为128的数组
    Queue<Character> queue = new LinkedList<Character>(); //新建一个队列
 
    //Insert one char from stringstream在字符串流中插入一个字符
    public void Insert(char ch) {
        if (charCnt[ch]++ == 0)   //新来的单身字符，入队
            queue.add(ch);
    }
    //return the first appearence once char in current stringstream一旦字符在字符串流中存在就返回第一次的数
    public char FirstAppearingOnce() {
        Character CHAR = null;
        char c = 0;
        while ((CHAR = queue.peek()) != null) {  如果队列顶端不为null
            c = CHAR.charValue();
            if (charCnt[c] == 1){  //判断是否脱单了，没脱单则输出
                return c;
            }else{ 
queue.remove(); //脱单了就移出队列，它不会再回来了
}
        }
        return '#'; //队空，返回#
    }
}
方法二：offer书上代码中occurrence[i]数组有两个作用，很是巧妙：
1.记录字符出现的次数；
2.记录字符出现的次数。
public class Solution { 
int[] count = new int[256]; // 字符出现的次数 
int[] index = new int[256]; // 字符出现的次数 
int number = 0; 

public void Insert(char ch) { 
count[ch]++; 
index[ch] = number++;
} 
public char FirstAppearingOnce() { 
int minIndex = number; 
char ch = '#'; 
for (int i = 0; i < 256; i++) { // !!! 
if (count[i] == 1 && index[i] < minIndex) { 
ch = (char) i; 
minIndex = index[i]; 
} 
} 
return ch; 
} 
}

*****55.给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
思路：1.判断链表中有环 -> 2.得到环中节点的数目 -> 3.找到环中的入口节点
/*
 public class ListNode {
    int val;
    ListNode next = null;
    ListNode(int val) {
        this.val = val;
    }
}
*/
public class Solution {
 
    public ListNode EntryNodeOfLoop(ListNode pHead)
    {
        if(pHead == null){ //特例情况，链表为null
            return null;
        }
        // 1.判断链表中有环
        ListNode l=pHead,r=pHead;  //分别定义l和r复制pHead的链表
        boolean flag = false;  //定义一个标识位
        while(r != null && r.next!=null){ //如果r不为null且下一个指针不为null
            l=l.next;   //l指向下一位，r则指向下两位
            r=r.next.next;
            if(l==r){   //两者相等即为找到了环
                flag=true;
                break;
            }
        }
        if(!flag){
            return null;
        }else{
            // 2.得到环中节点的数目
            int n=1;
            r=r.next; //链表r指向下一位，只要l和r不等，r就一直往下一位指，数目自加一
            while(l!=r){
                r=r.next;
                n++;
            }
            // 3.找到环中的入口节点
            l=r=pHead;
            for(int i=0;i<n;i++){
                r=r.next;
            }
            while(l!=r){ //当l和r两者相等时返回l的值，即入口节点，不等就都指向下一位
                l=l.next;
                r=r.next;
            }
            return l;
        }
    }
}
方法2：优先使用set或者map的方法来解决这道问题
import java.util.HashSet;//hashset的引入
public class Solution {
 
    public ListNode EntryNodeOfLoop(ListNode pHead)
    {
        HashSet<ListNode> help = new HashSet<>(); //HashSet的初始化
        ListNode pointer = pHead;
        while(pointer!= null){
            if(help.contains(pointer))//练习判断集合中是否包含某一个元素的contains方法
                return pointer;
            else
                help.add(pointer);/练习判断集合中增加一个元素的add方法
            pointer = pointer.next;
        }
        return null;
    }
}


***56.在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
思路：多次遍历，第一次遍历把重复的结点值存入 set 容器，第二次遍历，当结点值存储在 set 容器中，就删除结点
import java.util.*;
public class Solution {
    public ListNode deleteDuplication(ListNode pHead){
        if(pHead == null){ //特例情况
            return  null;
        }
        // 先找出相同结点，存入 set
        HashSet<Integer> set = new HashSet<>();
        ListNode pre = pHead; //pre指向当前结点，cur指向当前结点的下一结点
        ListNode cur = pHead.next;
        while(cur != null){  //链表没有循环结束且相邻结点相同重复就添加该结点值
            if(cur.val == pre.val){
                set.add(cur.val);
            }
            pre = cur;  //不相同就向下移动一位
            cur = cur.next;
        }
        // 再根据相同节点删除，这里的删除部分看不懂？？？
        // 先删头部  第一个相同的数
        while(pHead != null && set.contains(pHead.val)){
            pHead = pHead.next;
        }
        if(pHead == null){
            return null;
        }
        // 再删中间结点
        pre = pHead;
        cur = pHead.next;
        while(cur != null){
            if(set.contains(cur.val)){
                pre.next = cur.next;
                cur = cur.next;
            }else{
                pre = cur;
                cur = cur.next;
            }
        }
        return pHead;
    }
}
*****57.给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
思路：还原二叉树：既然给了二叉树的某个结点，且二叉树存储着指向父结点的指针（next），那我们可以先找到根节点，再对树进行中序遍历，最后根据中序遍历结果找到给定结点的下一结点
import java.util.*;
public class Solution {
    static ArrayList<TreeLinkNode> list = new ArrayList<>(); //新建静态数组list
    public TreeLinkNode GetNext(TreeLinkNode pNode){
        TreeLinkNode par = pNode;
        while(par.next != null){ //如果二叉树的下一结点不为null就指向下一结点
            par = par.next;
        }
        InOrder(par);
        for(int i=0;i<list.size();i++){ //遍历二叉树
            if(pNode == list.get(i)){  //如果得到和给定的结点值相等就返回i
                return i == list.size()-1?null:list.get(i+1);
            }
        }
        return null;
    }
    void InOrder(TreeLinkNode pNode){ //这里看不懂？？？
        if(pNode!=null){  //二叉树不为空就寻找左子树和右子树
            InOrder(pNode.left);
            list.add(pNode);
            InOrder(pNode.right);
        }
    }
}
58.请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
思路：就是按题意先画一棵“大”一点的对称二叉树，然后按树的一层一层比较一下，看看怎么算是满足对称的二叉树。
public class Solution {
    public boolean jude(TreeNode node1, TreeNode node2) {  //定义两颗二叉树node1和node2
        if (node1 == null && node2 == null) { //特例情况，两颗树均为null，返回真
            return true;
        } else if (node1 == null || node2 == null) { //两棵树有一颗为null就返回false
            return false;
        }
        if (node1.val != node2.val) { //如果两棵树的值不同也返回false
            return false;
        } else {  //两棵树的值相同时判断node1的左子树与node2的右子树是否相等且node1的右子树和node2的左子树是否相等，也就是镜像
            return jude(node1.left, node2.right) && jude(node1.right, node2.left);
        }
    }
    public boolean isSymmetrical(TreeNode pRoot) {
        return pRoot==null || jude(pRoot.left, pRoot.right); //pRoot为null时返回null，不为null返回jude
    }
}
*****59.请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
思路：
主要的方法与BFS写法没什么区别 
BFS里是每次只取一个，而我们这里先得到队列长度size，这个size就是这一层的节点个数，然后通过for循环去poll出这size个节点，这里和按行取值二叉树返回ArrayList<ArrayList<Integer>>这种题型的解法一样，之字形取值的核心思路就是通过两个方法：
list.add(T): 按照索引顺序从小到大依次添加 
list.add(index, T): 将元素插入index位置，index索引后的元素依次后移,这就完成了每一行元素的倒序，或者使用Collection.reverse()方法倒序也可以
import java.util.LinkedList;
public class Solution {
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        LinkedList<TreeNode> q = new LinkedList<>();
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        boolean rev = true;
        q.add(pRoot);
        while(!q.isEmpty()){ //如果二叉树q不为null
            int size = q.size();
            ArrayList<Integer> list = new ArrayList<>();
            for(int i=0; i<size; i++){ //遍历size
                TreeNode node = q.poll();
                if(node == null){continue;}
                if(rev){   //如果是真就添加node的值
                    list.add(node.val);
                }else{   //如果为假酒在第0位添加node的值，后面的元素依次后移
                    list.add(0, node.val);
                }
                q.offer(node.left); //遍历左右子树，这里的offer是什么作用？？？
                q.offer(node.right);
            }
            if(list.size()!=0){res.add(list);}
            rev=!rev;
        }
        return res;
    }
}
***60.从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。（和上一题方法类似）
思路：本应该是使用arraylist存储每一行节点，但是其实用queue存储之后整个程序的判断逻辑就清楚很多，因此数据结构活学活用，就非常好了。
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
/*
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;
    public TreeNode(int val) {
        this.val = val;
    }
}
*/
public class Solution {
    ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer> > result = new ArrayList<ArrayList<Integer> >();  
        if(pRoot != null){    //二叉树不为0
            Queue<TreeNode> up = new LinkedList<TreeNode>();//用来存储上一层节点
            Queue<TreeNode> down = new LinkedList<TreeNode>();//用来存储下一层节点
            up.offer(pRoot);
            while(!up.isEmpty()){  //up中不为null
                ArrayList<Integer> newLine = new ArrayList<Integer>();//新增一行
                while(!up.isEmpty()){
                    TreeNode temp = up.poll();
                    newLine.add(temp.val);  //在新增的行上添加二叉树temp的值
                    if(temp.left != null)  //如果二叉树temp左子树不为null就在down队列里存储左子树
                        down.offer(temp.left);
                    if(temp.right != null)//如果二叉树temp右子树不为null就在down队列里存储右子树
                        down.offer(temp.right);
                }
                result.add(newLine); //在结果上添加新增的行
                Queue<TreeNode> temp = up; //up和down互换位置
                up = down;
                down = temp;
            }
        }
        return result;
    }
}
*****61.请实现两个函数，分别用来序列化和反序列化二叉树
二叉树的序列化是指：把一棵二叉树按照某种遍历方式的结果以某种格式保存为字符串，从而使得内存中建立起来的二叉树可以持久保存。序列化可以基于先序、中序、后序、层序的二叉树遍历方式来进行修改，序列化的结果是一个字符串，序列化时通过 某种符号表示空节点（#），以 ！ 表示一个结点值的结束（value!）。
    二叉树的反序列化是指：根据某种遍历顺序得到的序列化字符串结果str，重构二叉树。
思路：（序列化容易理解，反序列化还要看）
序列化很简单，只需要在遇到null的时候添加#!号，遇到数值添加!即可
反序列化的时候，由于采用的是先序遍历，此时如果遇到了#号，我们知道左边结束了,要开启右边，如果再次遇到#,表示当前整个部分的左边结束了要开始右子树。。依次类推。
public class Solution{//序列化与反序列化均定义了一个私有类型的方法来实现
String Serialize(TreeNode root) { //序列化
    if (root == null) return ""; //特例情况
    return helpSerialize(root, new StringBuilder()).toString(); //返回字符串序列化的结果
}
private StringBuilder helpSerialize(TreeNode root, StringBuilder s) {
    if (root == null) return s; //如果二叉树root为null，直接返回s，并在结尾添加！
    s.append(root.val).append("!");
    if (root.left != null) { //如果二叉树root左子树不为null，就将左子树添加到s中
        helpSerialize(root.left, s);
    } else {    //左子树为null
        s.append("#!"); // 为null的话直接添加即可
    }
    if (root.right != null) { //如果二叉树root右子树不为null，就将右子树添加到s中
        helpSerialize(root.right, s);
    } else {  //右子树为null
        s.append("#!");
    }
    return s;
}
/*反序列化*/
private int index = 0; // 设置全局主要是遇到了#号的时候需要直接前进并返回null
TreeNode Deserialize(String str) {
    if (str == null || str.length() == 0) return null; //特例情况，字符串str为空
    String[] split = str.split("!");  //定义一个字符串split
    return helpDeserialize(split);
}
private TreeNode helpDeserialize(String[] strings) {
    if (strings[index].equals("#")) {  //如果字符串中的数值和#相等时，数据前移
        index++;// 数据前进
        return null;
    }
    // 当前值作为节点已经被用
    TreeNode root = new TreeNode(Integer.valueOf(strings[index]));
    index++; // index++到达下一个需要反序列化的值
    root.left = helpDeserialize(strings); //字符串值分别赋给二叉树root的左右子树
    root.right = helpDeserialize(strings);
    return root;  //返回二叉树
}
}
*****62.给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。
import java.util.Stack;
public class Solution {
    TreeNode KthNode(TreeNode pRoot, int k)
    {
        if(pRoot == null || k <= 0){ //特例情况
            return null;
        }
        Stack<TreeNode> stack = new Stack<>(); //建立栈stack
        TreeNode cur = pRoot;  //令cur等于二叉树pRoot
//while 部分为中序遍历
        while(!stack.isEmpty() || cur != null){  //栈不为null且二叉树不为null
            if(cur != null){
                stack.push(cur); //当前节点不为null，应该寻找左儿子
                cur = cur.left;
            }else{                //cur为null
                cur = stack.pop();//当前节点null则弹出栈内元素，相当于按顺序输出最小值。
                if(--k <= 0){ //计数器功能，这里的k值判定有些问题？？？
                    return cur;
                }
                cur = cur.right; //寻找右儿子
            }
        }
        return null;
    }
}
*****63.如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
思路：利用优先队列，优先队列分为大顶堆和小顶堆，默认维护的是小顶堆的优先队列
需要求的是中位数，如果我将 1 2 3 4 5 6 7 8定为最终的数据流
此时的中位数是4+5求均值。为什么是4，为什么是5
利用队列我们就可以看得很清楚，4是前半部分最大的值，肯定是维系在大顶堆
而5是后半部分的最小值，肯定是维系在小顶堆。
问题就好理解了：
使用小顶堆存大数据，使用大顶堆存小数据。这样堆顶一取出就是中位数了。
import java.util.PriorityQueue;
import java.util.Comparator;
public class Solution{
private int cnt = 0;  //定义数量，默认为0
private PriorityQueue<Integer> low = new PriorityQueue<>(); //定义一个优先队列low 小顶堆，用该堆记录位于中位数后面的部分
// 默认维护小顶堆
private PriorityQueue<Integer> high = new PriorityQueue<>(new Comparator<Integer>() { //大顶堆，用该堆记录位于中位数前面的部分
    @Override
    public int compare(Integer o1, Integer o2) { //比较函数，这里没看懂？？？比较o2和o1的大小
        return o2.compareTo(o1);
    }
});
public void Insert(Integer num) {  //读取数据流
    // 数量++
    cnt++;
    // 如果为奇数的话
    if ((cnt %2) == 1) {
        // 由于奇数，需要存放在大顶堆上
        // 但是呢，现在你不知道num与小顶堆的情况
        // 小顶堆存放的是后半段大的数
        // 如果当前值比小顶堆上的那个数更大
        if (!low.isEmpty() && num > low.peek()) { //如果小顶堆不为null且num大于小顶堆的那个数
            // 存进去
            low.offer(num);
            // 然后在将那个最小的吐出来
            num = low.poll();
        } // 最小的就放到大顶堆，因为它存放前半段
        high.offer(num);
    } else {
        // 偶数的话，此时需要存放的是小的数
        // 注意无论是大顶堆还是小顶堆，吐出数的前提是得有数
        if (!high.isEmpty() && num < high.peek()) { //如果大顶堆不为null且num小于大顶堆的那个数就存进去
            high.offer(num);
            num = high.poll();//再将最大的那个数吐出来
        } // 大数被吐出，小顶堆插入
        low.offer(num);
    }
}
public Double GetMedian() {// 表明是偶数，获取中位数
    double res = 0;
    // 奇数
    if ((cnt %2) == 1) {
        res = high.peek();
    } else { //偶数
        res = (high.peek() + low.peek()) / 2.0;
    }
    return res;
}
}
***64.给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。（移动窗口）
思路：用一个大顶堆，保存当前滑动窗口中的数据。滑动窗口每次移动一格，就将前面一个数出堆，后面一个数入堆。
import java.util.*;
public class Solution {
    public PriorityQueue<Integer> maxQueue = new PriorityQueue<Integer>((o1,o2)->o2-o1); //大顶堆
    public ArrayList<Integer> result = new ArrayList<Integer>();  //保存结果
    public ArrayList<Integer> maxInWindows(int [] num, int size)
    {
        if(num==null || num.length<=0 || size<=0 || size>num.length){//特例情况
            return result;
        }
        int count=0;  //计数count
        for(;count<size;count++){//初始化滑动窗口，遍历滑动
            maxQueue.offer(num[count]); //将窗口得到的数值放入大顶堆
        }
        while(count<num.length){//对每次操作，找到最大值（用优先队列的大顶堆），然后向后滑动（出堆一个，入堆一个）
            result.add(maxQueue.peek());  //将最大值放入result
            maxQueue.remove(num[count-size]); //大顶堆的前一个数字出堆
            maxQueue.add(num[count]);         //大顶堆的后一个数字入堆 
            count++;
        }
        result.add(maxQueue.peek());//最后一次入堆后没保存结果，这里额外做一次即可
        return result;
    }
}
*****65.请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如   矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
思路：回溯
基本思想： 
0.根据给定数组，初始化一个标志位数组，初始化为false，表示未走过，true表示已经走过，不能走第二次 
1.根据行数和列数，遍历数组，先找到一个与str字符串的第一个元素相匹配的矩阵元素，进入judge 
2.根据i和j先确定一维数组的位置，因为给定的matrix是一个一维数组 
3.确定递归终止条件：越界，当前找到的矩阵值不等于数组对应位置的值，已经走过的，这三类情况，都直接false，说明这条路不通 
4.若k，就是待判定的字符串str的索引已经判断到了最后一位，此时说明是匹配成功的 
5.下面就是本题的精髓，递归不断地寻找周围四个格子是否符合条件，只要有一个格子符合条件，就继续再找这个符合条件的格子的四周是否存在符合条件的格子，直到k到达末尾或者不满足递归条件就停止。 
6.走到这一步，说明本次是不成功的，我们要还原一下标志位数组index处的标志位，进入下一轮的判断。
public class Solution {
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str)
    {
        //标志位，初始化为false
        boolean[] flag = new boolean[matrix.length];
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                 //循环遍历二维数组，找到起点等于str第一个元素的值，再递归判断四周是否有符合条件的----回溯法
                 if(judge(matrix,i,j,rows,cols,flag,str,0)){
                     return true;
                 }
            }
        }
        return false;
    }
    //judge(初始矩阵，索引行坐标i，索引纵坐标j，矩阵行数，矩阵列数，待判断的字符串，字符串索引初始为0即先判断字符串的第一位)
    private boolean judge(char[] matrix,int i,int j,int rows,int cols,boolean[] flag,char[] str,int k){
        //先根据i和j计算匹配的第一个元素转为一维数组的位置
        int index = i*cols+j;
        //递归终止条件
        if(i<0 || j<0 || i>=rows || j>=cols || matrix[index] != str[k] || flag[index] == true)
            return false;
        //若k已经到达str末尾了，说明之前的都已经匹配成功了，直接返回true即可
        if(k == str.length-1)
            return true;
        //要走的第一个位置置为true，表示已经走过了
        flag[index] = true;
        //回溯，递归寻找，每次找到了就给k加一，找不到，还原，四个方向都寻找
        if(judge(matrix,i-1,j,rows,cols,flag,str,k+1) ||
           judge(matrix,i+1,j,rows,cols,flag,str,k+1) ||
           judge(matrix,i,j-1,rows,cols,flag,str,k+1) ||
           judge(matrix,i,j+1,rows,cols,flag,str,k+1)  )
        {
            return true;
        }
        //走到这，说明这一条路不通，还原，再试其他的路径
        flag[index] = false;
        return false;
    }
}
*******66.地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
public class Solution {      //这题整个看不懂？？？
    public int movingCount(int threshold, int rows, int cols) {
        int flag[][] = new int[rows][cols];   //记录是否已经走过
        return helper(0, 0, rows, cols, flag, threshold);
    }
    private int helper(int i, int j, int rows, int cols, int[][] flag, int threshold) {
        if (i < 0 || i >= rows || j < 0 || j >= cols || numSum(i) + numSum(j)  > threshold || flag[i][j] == 1) return 0;      //所有特例情况
        flag[i][j] = 1;
        return helper(i - 1, j, rows, cols, flag, threshold)    //寻找该点的上下左右四个格子能否移动到
            + helper(i + 1, j, rows, cols, flag, threshold)
            + helper(i, j - 1, rows, cols, flag, threshold) 
            + helper(i, j + 1, rows, cols, flag, threshold) 
            + 1;
    }
    private int numSum(int i) {
        int sum = 0;
        do{
            sum += i%10;
        }while((i = i/10) > 0);
        return sum;
    }
}
方法二：
public class Solution {
 // 判断坐标是否符合要求
 public boolean isValid(int row, int col, int threshold){
 int sum = 0; 
while(row > 0){ 
sum += row%10; 
row = row/10; 
} 
while(col >0){ 
sum += col%10;
col = col/10; 
} 
if(sum > threshold)return false;
 else return true; 
} 
//统计能够走到的次数 
public int count = 0;

 public void help(int i, int j, int threshold, int rows, int cols, int[][] flag){ 
if(i < 0 || i >= rows || j < 0 || j >= cols)return;//如果坐标不符合则不计数 
if(flag[i][j] == 1)return;//如果曾经被访问过则不计数 
if(!isValid(i, j, threshold)){ 
flag[i][j] = 1;//如果不满足坐标有效性，则不计数并且标记是访问的 
return;
 } 
//无论是广度优先遍历还是深度优先遍历，我们一定要知道的时候遍历一定会有终止条件也就是要能够停止，
 //不然程序就会陷入死循环，这个一定是我们做此类题目必须要注意的地方 
flag[i][j] = 1; 
count++;
 help(i-1, j, threshold, rows, cols, flag);//遍历上下左右节点 
help(i+1, j, threshold, rows, cols, flag); 
help(i, j-1, threshold, rows, cols, flag); 
help(i, j+1, threshold, rows, cols, flag); 
} 
public int movingCount(int threshold, int rows, int cols) { 
int[][] flag = new int[rows][cols];
 help(0, 0, threshold, rows, cols, flag);
 return count; 
} 
}


67.给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

方法一：时间复杂度O（n）
第一个循环从4开始，没啥问题吧。第二个循环为什么j<=i/2是因为1*3和3*1是一样的，没必要计算在内，只要计算到1*3和2*2就好了。然后就是取最大，1*3最大是3,2*2最大是4，那么dp[4]=res就是4，5,6,7，……，n一样的道理。
public class Solution {
    public int cutRope(int n) {
       // n<=3的情况，m>1必须要分段，例如：3必须分成1、2；1、1、1 ，n=3最大分段乘积是2,
        if(n==2)
            return 1;
        if(n==3)
            return 2;
        int[] dp = new int[n+1];
        /*
        下面3行是n>=4的情况，跟n<=3不同，4可以分很多段，比如分成1、3，
        这里的3可以不需要再分了，因为3分段最大才2，不分就是3。记录最大的。
         */
        dp[1]=1;
        dp[2]=2;
        dp[3]=3;
        int res=0;//记录最大的
        for (int i = 4; i <= n; i++) {  //因为1，2，3已经固定，所以从第四位开始遍历
            for (int j = 1; j <=i/2 ; j++) {  //j<=i/2是因为1*3和3*1是一致的，删除一种，取最大值即可
                res=Math.max(res,dp[j]*dp[i-j]);
            }
            dp[i]=res;
        }
        return dp[n];
    }
}
方法二：递归求解，但是复杂度会高很多
public class Solution {  //没考虑特例情况
    public int cutRope(int target) {
       return cutRope(target, 0);
    }
    public int cutRope(int target, int max) {
        int maxValue = max;
        for(int i = 1; i < target; ++i){  //遍历
            maxValue = Math.max(maxValue, i*cutRope(target -i, target -i));
        }
        return maxValue;
    }
}                                                                          
