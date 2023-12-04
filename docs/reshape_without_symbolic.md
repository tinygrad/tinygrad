## ["View.reshape without symbolic"](https://github.com/tinygrad/tinygrad/pull/2218)

This section contains the sketch proof of "Complete, Fast and Correct View.reshapes without using Symbolic". The goal is to reduce multi-views which cost runtime.

1. **old_shape = (s<sub>1</sub>,s<sub>2</sub>,...,s<sub>i</sub>,s<sub>(i+1)</sub>,...,s<sub>n</sub>)**
2. **old_stride = (st<sub>1</sub>, st<sub>2</sub>, ... ,st<sub>i</sub>, st<sub>(i+1)</sub>, ..., st<sub>n</sub>)**
3. **merge_old_shape = (p<sub>1</sub>, p<sub>2</sub>), where p<sub>1</sub> = s<sub>1</sub> * ... * s<sub>i</sub> & p<sub>2</sub> = s<sub>(i+1)</sub> * ... * s<sub>n</sub>**,
4. **new_shape = (k<sub>1</sub>, ..., k<sub>p</sub>, k<sub>(p+1)</sub>, ..., k<sub>l</sub>)**
5. **prod(new_shape) = p<sub>1</sub> * p<sub>2</sub>** (trivial)
6. **mask** and **new_mask** represent valid indexes before & after reshape respectively.
 
 
### Assumption

**p<sub>1</sub>** & **p<sub>2</sub>** individually are mergeable (we will discuss later on this) & we cannot merge **p<sub>1</sub>** & **p<sub>2</sub>**.

### Claim

If **prod([k<sub>1</sub> ... k<sub>p</sub>]) < p<sub>1</sub>** and **prod([k<sub>1</sub> ... k<sub>(p+1)</sub>]) > p<sub>1</sub>**, reshape is not possible.

**Proof**

**k<sub>(p+1)</sub>** will require some dimensions from **p<sub>1</sub>** & some from **p<sub>2</sub>**, which means **p<sub>1</sub>** & **p<sub>2</sub>** should be mergeable, but they are not.

**Conclusion**

Hence, reshape is only possible **if ∃ a p, where prod([k<sub>1</sub> .. k<sub>p</sub>]) = p<sub>1</sub>**.


### Conditions for mergeability

**Case 1 - All non-zero strides**

They will merge **if st<sub>x</sub> = st<sub>(x+1)</sub> * s<sub>(x+1)</sub>, where x ∈ [1, ..., i-1, i+1, ..., n-1]**.

**Proof**

Lets consider merging of **(s<sub>1</sub> ... s<sub>i</sub>) -> p<sub>1</sub>**, here we have to get a single new stride corresponding to **p<sub>1</sub>**. For which it has to be contiguous. 

**Case 2 - Some stride is zero**

Let **st<sub>j</sub> = 0 & st<sub>(j+1)</sub> != 0 & s<sub>(j+1)</sub> > 1, where 1 < j < i**.

If **s<sub>j</sub> = 1** , reshape is trivial.

If **s<sub>j</sub> > 1**,
- If **mask<sub>j</sub>** has range > 1,
	reshape is not possible, because **s<sub>(j+1)</sub>** will need to be repeated at-least once and a single stride can't capture repetition.
- If **mask<sub>j</sub>** has range = 1,  reshape is possible, since it is virtually shape = 1, with some offset.



### Conditions for reshaping mask

**Case 1 - Splitting Dimension** - Mask shouldn't be cut for successful reshape.

- **Example** - 
[1,2,3,4,5,6,7,8] -> [[1,2,3,4], [5,6,7,8]] ; **mask** = ((2,6)) ; **new_mask[0]** = (0,2) (trivial split).

- **new_mask[1]** = not possible. It is only possible if **mask spans [1-8] or lies within a single dimension [1-4] or [5-8]**.


**Case 2 - Combining Dimension** - Mask should unfold continuously.

- **Example** - **[[1,2],[3,4],[5,6]] -> [1,2,3,4,5,6]**;  **mask** = ((0,2),(0,2)).

- **new_mask** = (0,4); only possible because **mask<sub>1</sub>** span the whole dimension.

- If **mask<sub>1</sub>** did not span the whole dimension, the only way combining would be possible is if **mask<sub>0</sub>** had range 1 as shown below.
	- **[[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]**; **mask** = ((1,2),(0,2)); **new_mask** = ((3,5))