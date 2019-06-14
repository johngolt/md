###### 数组 Array

Given $n$ non-negative integers representing an elevation map where the width of each bar is $1$, compute how much water it is able to trap after raining.

解法：$\min(\text{max_left}[i],\text{max_right}[i]) - \text{height[i]}$

![](./picture/1/8.png)

Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump length at that position. Your goal is to reach the last index in the minimum number of jumps.