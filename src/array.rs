#[cfg(feature = "nightly")]
use alloc::alloc::{Allocator, Global};
use alloc::collections::TryReserveError;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::borrow::{Borrow, BorrowMut};
use core::fmt::{self, Debug, Formatter};
use core::hash::{Hash, Hasher};
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ops::{Deref, DerefMut, Index, IndexMut, RangeBounds};
use core::{ptr, slice};

#[cfg(not(feature = "nightly"))]
use crate::allocator::{Allocator, Global};
use crate::buffer::{Buffer, Drain, DynBuffer, Owned};
use crate::dim::{Const, Dim, Dyn};
use crate::expr::{self, IntoExpr, Iter, Map, Zip};
use crate::expr::{Apply, Expand, Expression, FromExpression, IntoExpression};
use crate::index::SliceIndex;
use crate::layout::{Dense, Layout};
use crate::mapping::{DenseMapping, Mapping};
use crate::shape::{ConstShape, DynRank, IntoShape, Rank, Shape};
use crate::slice::Slice;
use crate::traits::IntoCloned;
use crate::view::{View, ViewMut};

#[cfg(not(feature = "nightly"))]
macro_rules! vec_t {
    ($type:ty, $alloc:ty) => {
        Vec<$type>
    };
}

#[cfg(feature = "nightly")]
macro_rules! vec_t {
    ($type:ty, $alloc:ty) => {
        Vec<$type, $alloc>
    };
}

/// Dense multidimensional array.
#[repr(transparent)]
pub struct Array<T, S: Shape = DynRank, A: Allocator = Global> {
    buffer: S::Buffer<T, A>,
}

/// Multidimensional array with dynamically-sized dimensions and dense layout.
pub type DArray<T, const N: usize, A = Global> = Array<T, Rank<N>, A>;

/// Dense multidimensional array.
///
/// This type alias is for backward compatibility, use `Array` instead.
pub type Tensor<T, S = DynRank, A = Global> = Array<T, S, A>;

/// Multidimensional array with dynamically-sized dimensions and dense layout.
///
/// This type alias is for backward compatibility, use `DArray` instead.
pub type DTensor<T, const N: usize, A = Global> = Array<T, Rank<N>, A>;

impl<T, S: Shape, A: Allocator> Array<T, S, A> {
    /// Returns a reference to the underlying allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn allocator(&self) -> &A {
        self.buffer.allocator()
    }

    /// Creates an array from the given element with the specified allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn from_elem_in<I: IntoShape<IntoShape = S>>(shape: I, elem: T, alloc: A) -> Self
    where
        T: Clone,
    {
        Self::from_expr_in(expr::from_elem(shape, elem), alloc)
    }

    /// Creates an array from an expression with the specified allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn from_expr_in<I: IntoExpression<Item = T, Shape = S>>(expr: I, alloc: A) -> Self {
        Self::with_expr_in(expr.into_expr(), alloc)
    }

    /// Creates an array with the results from the given function and the specified allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn from_fn_in<I: IntoShape<IntoShape = S>, F>(shape: I, f: F, alloc: A) -> Self
    where
        F: FnMut(&[usize]) -> T,
    {
        Self::from_expr_in(expr::from_fn(shape, f), alloc)
    }

    /// Converts the array into a new array with the given shape type.
    ///
    /// # Panics
    ///
    /// Panics if the shape is not matching static rank or constant-sized dimensions.
    #[inline]
    pub fn into_buffer<R: Shape>(self) -> Array<T, R, A> {
        Array { buffer: self.buffer.into_buffer() }
    }

    /// Converts the array into an array with dynamic rank.
    #[inline]
    pub fn into_dyn(self) -> Array<T, DynRank, A> {
        self.into_buffer()
    }

    /// Converts the array into a one-dimensional array.
    #[cfg(not(feature = "nightly"))]
    #[inline]
    pub fn into_flat(self) -> Array<T, (Dyn,), A> {
        let vec = self.into_vec();
        let shape = (vec.len(),);

        Array { buffer: unsafe { DynBuffer::from_parts(vec, shape) } }
    }

    /// Converts the array into a one-dimensional array.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn into_flat(self) -> Array<T, (Dyn,), A> {
        self.into_vec().into()
    }

    /// Converts the array into a new array with the given shape type.
    ///
    /// This method is deprecated, use `into_buffer` instead.
    #[deprecated]
    #[inline]
    pub fn into_mapping<R: Shape>(self) -> Array<T, R, A> {
        self.into_buffer()
    }

    /// Converts an array with a single element into the contained value.
    ///
    /// # Panics
    ///
    /// Panics if the array length is not equal to one.
    #[inline]
    pub fn into_scalar(self) -> T {
        assert!(self.len() == 1, "invalid length");

        self.into_iter().next().unwrap()
    }

    /// Converts the array into a reshaped array, which must have the same length.
    ///
    /// At most one dimension can have dynamic size `usize::MAX`, and is then inferred
    /// from the other dimensions and the array length.
    ///
    /// # Examples
    ///
    /// ```
    /// use mdarray::{darray, view};
    ///
    /// let a = darray![[1, 2, 3], [4, 5, 6]];
    ///
    /// assert_eq!(a.into_shape([!0, 2]), view![[1, 2], [3, 4], [5, 6]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed.
    #[inline]
    pub fn into_shape<I: IntoShape>(self, shape: I) -> Array<T, I::IntoShape, A> {
        Array { buffer: self.buffer.into_shape::<I::IntoShape>(shape.into_shape()) }
    }

    /// Converts the array into a vector.
    #[inline]
    pub fn into_vec(self) -> vec_t!(T, A) {
        let len = self.len();
        let (vec, _) = self.into_shape(len).buffer.into_parts();

        vec
    }

    /// Returns an array with the same shape, and the given closure applied to each element.
    ///
    /// The input array is reused if the memory layout for the input and output elements
    /// are the same, or otherwise a new array is allocated for the result.
    #[inline]
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> Array<U, S, A> {
        self.zip_with(expr::fill(()), |(x, ())| f(x))
    }

    /// Creates an array with uninitialized elements and the specified allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn uninit_in<I: IntoShape<IntoShape = S>>(
        shape: I,
        alloc: A,
    ) -> Array<MaybeUninit<T>, S, A> {
        Array { buffer: <S::Buffer<T, A>>::uninit_in(shape.into_shape(), alloc) }
    }

    /// Creates an array with elements set to zero and the specified allocator.
    ///
    /// Zero elements are created using `Default::default()`.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn zeros_in<I: IntoShape<IntoShape = S>>(shape: I, alloc: A) -> Self
    where
        T: Default,
    {
        Self::from_expr_in(expr::from_elem(shape, ()).map(|_| T::default()), alloc)
    }

    #[inline]
    pub(crate) fn into_inner(self) -> S::Buffer<T, A> {
        self.buffer
    }

    #[inline]
    pub(crate) fn with_expr_in<E: Expression<Item = T>>(expr: E, alloc: A) -> Self {
        struct DropGuard<'a, T, S: Shape, A: Allocator> {
            array: &'a mut Array<MaybeUninit<T>, S, A>,
            index: usize,
        }

        impl<T, S: Shape, A: Allocator> Drop for DropGuard<'_, T, S, A> {
            #[inline]
            fn drop(&mut self) {
                let ptr = self.array.as_mut_ptr() as *mut T;

                unsafe {
                    ptr::slice_from_raw_parts_mut(ptr, self.index).drop_in_place();
                }
            }
        }

        let shape = expr.shape().with_dims(S::from_dims);

        #[cfg(not(feature = "nightly"))]
        let mut array = Array { buffer: <S::Buffer<T, A>>::uninit_in(shape, alloc) };
        #[cfg(feature = "nightly")]
        let mut array = Array::uninit_in(shape, alloc);
        let mut guard = DropGuard { array: &mut array, index: 0 };

        let expr = guard.array.expr_mut().zip(expr);

        expr.for_each(|(x, y)| {
            _ = x.write(y);
            guard.index += 1;
        });

        mem::forget(guard);

        unsafe { array.assume_init() }
    }
}

impl<T, S: Shape<Buffer<T, A> = DynBuffer<T, S, A>>, A: Allocator> Array<T, S, A> {
    /// Moves all elements from another array into the array along the first dimension.
    ///
    /// If the array is empty, it is reshaped to match the shape of the other array.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match, if the rank is not the same and
    /// at least 1, or if the first dimension is not dynamically-sized.
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        self.expand(other.drain(..));
    }

    /// Returns the number of elements the array can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Clears the array, removing all values.
    ///
    /// If the array type has dynamic rank, the rank is set to 1.
    ///
    /// Note that this method has no effect on the allocated capacity of the array.
    #[inline]
    pub fn clear(&mut self) {
        unsafe {
            self.buffer.with_mut_parts(|vec, shape| {
                vec.clear();
                *shape = S::default();
            });
        }
    }

    /// Removes the specified range from the array along the first dimension,
    /// and returns the removed range as an expression.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1, or if the first dimension
    /// is not dynamically-sized.
    #[inline]
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> IntoExpr<T, Drain<'_, T, S, A>> {
        assert!(self.rank() > 0, "invalid rank");
        assert!(S::Head::SIZE.is_none(), "first dimension not dynamically-sized");

        #[cfg(not(feature = "nightly"))]
        let range = crate::index::range(range, ..self.dim(0));
        #[cfg(feature = "nightly")]
        let range = slice::range(range, ..self.dim(0));

        IntoExpr::new(Drain::new(self, range.start, range.end))
    }

    /// Appends an expression to the array along the first dimension with broadcasting,
    /// cloning elements if needed.
    ///
    /// If the array is empty, it is reshaped to match the shape of the expression.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match, if the rank is not the same and
    /// at least 1, or if the first dimension is not dynamically-sized.
    #[inline]
    pub fn expand<I: IntoExpression<Item: IntoCloned<T>>>(&mut self, expr: I) {
        assert!(self.rank() > 0, "invalid rank");
        assert!(S::Head::SIZE.is_none(), "first dimension not dynamically-sized");

        let expr = expr.into_expr();
        let len = expr.len();

        if len > 0 {
            unsafe {
                self.buffer.with_mut_parts(|vec, shape| {
                    vec.reserve(len);

                    expr.shape().with_dims(|src| {
                        if shape.is_empty() {
                            if src.len() == shape.rank() {
                                shape.with_mut_dims(|dims| dims.copy_from_slice(src));
                            } else {
                                *shape = Shape::from_dims(src);
                            }
                        } else {
                            shape.with_mut_dims(|dims| {
                                assert!(src.len() == dims.len(), "invalid rank");
                                assert!(src[1..] == dims[1..], "inner dimensions mismatch");

                                dims[0] += src[0];
                            });
                        }
                    });

                    expr.clone_into_vec(vec);
                });
            }
        }
    }

    /// Creates an array from raw components of another array with the specified allocator.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape, capacity and allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub unsafe fn from_raw_parts_in(ptr: *mut T, shape: S, capacity: usize, alloc: A) -> Self {
        unsafe {
            let vec = Vec::from_raw_parts_in(ptr, shape.len(), capacity, alloc);

            Self { buffer: DynBuffer::from_parts(vec, shape) }
        }
    }

    /// Decomposes an array into its raw components including the allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn into_raw_parts_with_alloc(self) -> (*mut T, S, usize, A) {
        let (vec, shape) = self.buffer.into_parts();
        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_alloc();

        (ptr, shape, capacity, alloc)
    }

    /// Creates a new, empty array with the specified allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn new_in(alloc: A) -> Self {
        assert!(S::default().checked_len().is_some(), "invalid length");

        Self { buffer: unsafe { DynBuffer::from_parts(Vec::new_in(alloc), S::default()) } }
    }

    /// Reserves capacity for at least the additional number of elements in the array.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            self.buffer.with_mut_parts(|vec, _| vec.reserve(additional));
        }
    }

    /// Reserves the minimum capacity for the additional number of elements in the array.
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        unsafe {
            self.buffer.with_mut_parts(|vec, _| vec.reserve_exact(additional));
        }
    }

    /// Resizes the array to the new shape, creating new elements with the given value.
    #[inline]
    pub fn resize<I: IntoShape<IntoShape = S>>(&mut self, new_shape: I, value: T)
    where
        T: Clone,
    {
        new_shape.into_dims(|new_dims| self.buffer.resize_with(new_dims, || value.clone()));
    }

    /// Resizes the array to the new shape, creating new elements from the given closure.
    #[inline]
    pub fn resize_with<I: IntoShape<IntoShape = S>, F>(&mut self, new_shape: I, f: F)
    where
        F: FnMut() -> T,
    {
        new_shape.into_dims(|new_dims| self.buffer.resize_with(new_dims, f));
    }

    /// Forces the array layout mapping to the new mapping.
    ///
    /// This method is deprecated, use `set_shape` instead.
    #[allow(clippy::missing_safety_doc)]
    #[deprecated]
    #[inline]
    pub unsafe fn set_mapping(&mut self, new_mapping: DenseMapping<S>) {
        unsafe {
            self.set_shape(new_mapping.shape().clone());
        }
    }

    /// Forces the array layout mapping to the new shape.
    ///
    /// # Safety
    ///
    /// All elements within the array length must be initialized.
    #[inline]
    pub unsafe fn set_shape(&mut self, new_shape: S) {
        unsafe {
            self.buffer.set_shape(new_shape);
        }
    }

    /// Shrinks the capacity of the array with a lower bound.
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        unsafe {
            self.buffer.with_mut_parts(|vec, _| vec.shrink_to(min_capacity));
        }
    }

    /// Shrinks the capacity of the array as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        unsafe {
            self.buffer.with_mut_parts(|vec, _| vec.shrink_to_fit());
        }
    }

    /// Returns the remaining spare capacity of the array as a slice of `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the array with data, before marking
    /// the data as initialized using the `set_shape` method.
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let ptr = self.as_mut_ptr();
        let len = self.capacity() - self.len();

        unsafe { slice::from_raw_parts_mut(ptr.add(self.len()).cast(), len) }
    }

    /// Shortens the array along the first dimension, keeping the first `size` indices.
    ///
    /// If `size` is greater or equal to the current dimension size, this has no effect.
    ///
    /// Note that this method has no effect on the allocated capacity of the array.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1, or if the first dimension
    /// is not dynamically-sized.
    #[inline]
    pub fn truncate(&mut self, size: usize) {
        assert!(self.rank() > 0, "invalid rank");
        assert!(S::Head::SIZE.is_none(), "first dimension not dynamically-sized");

        if size < self.dim(0) {
            unsafe {
                self.buffer.with_mut_parts(|vec, shape| {
                    shape.with_mut_dims(|dims| dims[0] = size);
                    vec.truncate(shape.len());
                });
            }
        }
    }

    /// Tries to reserve capacity for at least the additional number of elements in the array.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.buffer.with_mut_parts(|vec, _| vec.try_reserve(additional)) }
    }

    /// Tries to reserve the minimum capacity for the additional number of elements in the array.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.buffer.with_mut_parts(|vec, _| vec.try_reserve_exact(additional)) }
    }

    /// Creates a new, empty array with the specified capacity and allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        assert!(S::default().checked_len().is_some(), "invalid length");

        let vec = Vec::with_capacity_in(capacity, alloc);

        Self { buffer: unsafe { DynBuffer::from_parts(vec, S::default()) } }
    }
}

#[cfg(not(feature = "nightly"))]
impl<T, S: Shape> Array<T, S> {
    /// Creates an array from the given element.
    #[inline]
    pub fn from_elem<I: IntoShape<IntoShape = S>>(shape: I, elem: T) -> Self
    where
        T: Clone,
    {
        Self::from_expr(expr::from_elem(shape, elem))
    }

    /// Creates an array from an expression.
    #[inline]
    pub fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self {
        Self::with_expr_in(expr.into_expr(), Global)
    }

    /// Creates an array with the results from the given function.
    #[inline]
    pub fn from_fn<I: IntoShape<IntoShape = S>, F>(shape: I, f: F) -> Self
    where
        F: FnMut(&[usize]) -> T,
    {
        Self::from_expr(expr::from_fn(shape, f))
    }

    /// Creates an array with uninitialized elements.
    #[inline]
    pub fn uninit<I: IntoShape<IntoShape = S>>(shape: I) -> Array<MaybeUninit<T>, S> {
        Array { buffer: <S::Buffer<T, Global>>::uninit_in(shape.into_shape(), Global) }
    }

    /// Creates an array with elements set to zero.
    ///
    /// Zero elements are created using `Default::default()`.
    #[inline]
    pub fn zeros<I: IntoShape<IntoShape = S>>(shape: I) -> Self
    where
        T: Default,
    {
        Self::from_expr(expr::from_elem(shape, ()).map(|_| T::default()))
    }

    #[inline]
    pub(crate) fn clone_from_slice(&mut self, slice: &Slice<T, S>)
    where
        T: Clone,
    {
        self.buffer.clone_from_slice(slice);
    }
}

#[cfg(not(feature = "nightly"))]
impl<T, S: Shape<Buffer<T, Global> = DynBuffer<T, S>>> Array<T, S> {
    /// Creates an array from raw components of another array.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape and capacity.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, shape: S, capacity: usize) -> Self {
        unsafe {
            let vec = Vec::from_raw_parts(ptr, shape.len(), capacity);

            Self { buffer: DynBuffer::from_parts(vec, shape) }
        }
    }

    /// Decomposes an array into its raw components.
    #[inline]
    pub fn into_raw_parts(self) -> (*mut T, S, usize) {
        let (vec, shape) = self.buffer.into_parts();
        let mut vec = mem::ManuallyDrop::new(vec);

        (vec.as_mut_ptr(), shape, vec.capacity())
    }

    /// Creates a new, empty array.
    #[inline]
    pub fn new() -> Self {
        assert!(S::default().checked_len().is_some(), "invalid length");

        Self { buffer: unsafe { DynBuffer::from_parts(Vec::new(), S::default()) } }
    }

    /// Creates a new, empty array with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(S::default().checked_len().is_some(), "invalid length");

        let vec = Vec::with_capacity(capacity);

        Self { buffer: unsafe { DynBuffer::from_parts(vec, S::default()) } }
    }
}

#[cfg(feature = "nightly")]
impl<T, S: Shape> Array<T, S> {
    /// Creates an array from the given element.
    #[inline]
    pub fn from_elem<I: IntoShape<IntoShape = S>>(shape: I, elem: T) -> Self
    where
        T: Clone,
    {
        Self::from_elem_in(shape, elem, Global)
    }

    /// Creates an array from an expression.
    #[inline]
    pub fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self {
        Self::from_expr_in(expr, Global)
    }

    /// Creates an array with the results from the given function.
    #[inline]
    pub fn from_fn<I: IntoShape<IntoShape = S>, F>(shape: I, f: F) -> Self
    where
        F: FnMut(&[usize]) -> T,
    {
        Self::from_fn_in(shape, f, Global)
    }

    /// Creates an array with uninitialized elements.
    #[inline]
    pub fn uninit<I: IntoShape<IntoShape = S>>(shape: I) -> Array<MaybeUninit<T>, S> {
        Self::uninit_in(shape, Global)
    }

    /// Creates an array with elements set to zero.
    ///
    /// Zero elements are created using `Default::default()`.
    #[inline]
    pub fn zeros<I: IntoShape<IntoShape = S>>(shape: I) -> Self
    where
        T: Default,
    {
        Self::zeros_in(shape, Global)
    }

    #[inline]
    pub(crate) fn clone_from_slice(&mut self, slice: &Slice<T, S>)
    where
        T: Clone,
    {
        self.buffer.clone_from_slice(slice);
    }
}

#[cfg(feature = "nightly")]
impl<T, S: Shape<Buffer<T, Global> = DynBuffer<T, S>>> Array<T, S> {
    /// Creates an array from raw components of another array.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape and capacity.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, shape: S, capacity: usize) -> Self {
        unsafe { Self::from_raw_parts_in(ptr, shape, capacity, Global) }
    }

    /// Decomposes an array into its raw components.
    #[inline]
    pub fn into_raw_parts(self) -> (*mut T, S, usize) {
        let (ptr, shape, capacity, _) = self.into_raw_parts_with_alloc();

        (ptr, shape, capacity)
    }

    /// Creates a new, empty array.
    #[inline]
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a new, empty array with the specified capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<T, S: ConstShape> Array<T, S> {
    /// Creates an array from the given value with constant-sized dimensions.
    #[inline]
    pub fn fill(value: T) -> Self
    where
        T: Clone,
    {
        Self::with_expr_in(expr::from_elem(S::default(), value), Global)
    }

    /// Creates an array with the results from the given function and constant-sized dimensions.
    #[inline]
    pub fn fill_with<F: FnMut() -> T>(mut f: F) -> Self {
        Self::with_expr_in(expr::from_elem(S::default(), ()).map(|_| f()), Global)
    }
}

impl<T, S: Shape, A: Allocator> Array<MaybeUninit<T>, S, A> {
    /// Converts the array element type from `MaybeUninit<T>` to `T`.
    ///
    /// # Safety
    ///
    /// All elements in the array must be initialized, or the behavior is undefined.
    #[inline]
    pub unsafe fn assume_init(self) -> Array<T, S, A> {
        Array { buffer: unsafe { self.buffer.cast() } }
    }
}

impl<'a, T, U, S: Shape, A: Allocator> Apply<U> for &'a Array<T, S, A> {
    type Output<F: FnMut(&'a T) -> U> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a T, I::Item)) -> U> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    #[inline]
    fn apply<F: FnMut(&'a T) -> U>(self, f: F) -> Self::Output<F> {
        self.expr().map(f)
    }

    #[inline]
    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a T, I::Item)) -> U,
    {
        self.expr().zip(expr).map(f)
    }
}

impl<'a, T, U, S: Shape, A: Allocator> Apply<U> for &'a mut Array<T, S, A> {
    type Output<F: FnMut(&'a mut T) -> U> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a mut T, I::Item)) -> U> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    #[inline]
    fn apply<F: FnMut(&'a mut T) -> U>(self, f: F) -> Self::Output<F> {
        self.expr_mut().map(f)
    }

    #[inline]
    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a mut T, I::Item)) -> U,
    {
        self.expr_mut().zip(expr).map(f)
    }
}

impl<T, U, S: Shape, A: Allocator> Apply<U> for Array<T, S, A> {
    type Output<F: FnMut(T) -> U> = Array<U, S, A>;
    type ZippedWith<I: IntoExpression, F: FnMut((T, I::Item)) -> U> = Array<U, S, A>;

    #[inline]
    fn apply<F: FnMut(T) -> U>(self, f: F) -> Array<U, S, A> {
        self.map(f)
    }

    #[inline]
    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Array<U, S, A>
    where
        F: FnMut((T, I::Item)) -> U,
    {
        Array { buffer: self.buffer.zip_with(expr.into_expr(), f) }
    }
}

impl<T, S: Shape, A: Allocator> AsMut<Self> for Array<T, S, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<T, U: ?Sized, S: Shape, A: Allocator> AsMut<U> for Array<T, S, A>
where
    Slice<T, S>: AsMut<U>,
{
    #[inline]
    fn as_mut(&mut self) -> &mut U {
        (**self).as_mut()
    }
}

impl<T, S: Shape, A: Allocator> AsRef<Self> for Array<T, S, A> {
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<T, U: ?Sized, S: Shape, A: Allocator> AsRef<U> for Array<T, S, A>
where
    Slice<T, S>: AsRef<U>,
{
    #[inline]
    fn as_ref(&self) -> &U {
        (**self).as_ref()
    }
}

//
// The below implementations assume that casting a pointer from a primitive array to an
// Array is possible. This is ensured by the representations of Array and StaticBuffer.
//

macro_rules! impl_as_mut_ref {
    (($($xyz:tt),+), $array:tt) => {
        impl<T, $(const $xyz: usize),+> AsMut<Array<T, ($(Const<$xyz>,)+)>> for $array {
            #[inline]
            fn as_mut(&mut self) -> &mut Array<T, ($(Const<$xyz>,)+)> {
                unsafe { &mut *(self as *mut Self as *mut Array<T, ($(Const<$xyz>,)+)>) }
            }
        }

        impl<T, $(const $xyz: usize),+> AsRef<Array<T, ($(Const<$xyz>,)+)>> for $array {
            #[inline]
            fn as_ref(&self) -> &Array<T, ($(Const<$xyz>,)+)> {
                unsafe { &*(self as *const Self as *const Array<T, ($(Const<$xyz>,)+)>) }
            }
        }
    };
}

impl_as_mut_ref!((X), [T; X]);
impl_as_mut_ref!((X, Y), [[T; Y]; X]);
impl_as_mut_ref!((X, Y, Z), [[[T; Z]; Y]; X]);
impl_as_mut_ref!((X, Y, Z, W), [[[[T; W]; Z]; Y]; X]);
impl_as_mut_ref!((X, Y, Z, W, U), [[[[[T; U]; W]; Z]; Y]; X]);
impl_as_mut_ref!((X, Y, Z, W, U, V), [[[[[[T; V]; U]; W]; Z]; Y]; X]);

impl<T, S: Shape, A: Allocator> Borrow<Slice<T, S>> for Array<T, S, A> {
    #[inline]
    fn borrow(&self) -> &Slice<T, S> {
        self
    }
}

impl<T, S: Shape, A: Allocator> BorrowMut<Slice<T, S>> for Array<T, S, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut Slice<T, S> {
        self
    }
}

impl<T: Clone, S: Shape, A: Allocator + Clone> Clone for Array<T, S, A> {
    #[inline]
    fn clone(&self) -> Self {
        Array { buffer: self.buffer.clone() }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.buffer.clone_from(&source.buffer);
    }
}

impl<T: Copy, S: Shape<Buffer<T, A>: Copy>, A: Allocator + Copy> Copy for Array<T, S, A> {}

impl<T: Debug, S: Shape, A: Allocator> Debug for Array<T, S, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: Default, S: Shape> Default for Array<T, S> {
    #[inline]
    fn default() -> Self {
        Self::zeros(S::default())
    }
}

impl<T, S: Shape, A: Allocator> Deref for Array<T, S, A> {
    type Target = Slice<T, S>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.buffer.as_slice()
    }
}

impl<T, S: Shape, A: Allocator> DerefMut for Array<T, S, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut_slice()
    }
}

impl<T, U: IntoCloned<T>, S: Shape<Buffer<T, A> = DynBuffer<T, S, A>>, A: Allocator> Expand<U>
    for Array<T, S, A>
{
    #[inline]
    fn expand<I: IntoExpression<Item = U>>(&mut self, expr: I) {
        self.expand(expr);
    }
}

impl<'a, T: Clone, A: Allocator> Extend<&'a T> for Array<T, (Dyn,), A> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}

impl<T, A: Allocator> Extend<T> for Array<T, (Dyn,), A> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        unsafe {
            self.buffer.with_mut_parts(|vec, shape| {
                vec.extend(iter);
                *shape = (vec.len(),);
            });
        }
    }
}

impl<T: Clone> From<&[T]> for Array<T, (Dyn,)> {
    #[inline]
    fn from(value: &[T]) -> Self {
        Self::from(value.to_vec())
    }
}

impl<'a, T: 'a + Clone, S: Shape, L: Layout, I: IntoExpression<IntoExpr = View<'a, T, S, L>>>
    From<I> for Array<T, S>
{
    #[inline]
    fn from(value: I) -> Self {
        Self::from_expr(value.into_expr().cloned())
    }
}

impl<T, D: Dim, A: Allocator> From<Array<T, (D,), A>> for vec_t!(T, A) {
    #[inline]
    fn from(value: Array<T, (D,), A>) -> Self {
        value.into_vec()
    }
}

#[cfg(not(feature = "nightly"))]
impl<T> From<Vec<T>> for Array<T, (Dyn,)> {
    #[inline]
    fn from(value: Vec<T>) -> Self {
        let shape = (value.len(),);

        Self { buffer: unsafe { DynBuffer::from_parts(value, shape) } }
    }
}

#[cfg(feature = "nightly")]
impl<T, A: Allocator> From<Vec<T, A>> for Array<T, (Dyn,), A> {
    #[inline]
    fn from(value: Vec<T, A>) -> Self {
        let shape = (value.len(),);

        Self { buffer: unsafe { DynBuffer::from_parts(value, shape) } }
    }
}

macro_rules! impl_from_array {
    (($($xyz:tt),+), ($($abc:tt),+), $array:tt) => {
        impl<T: Clone $(,$xyz: Dim + From<Const<$abc>>)+ $(,const $abc: usize)+> From<&$array>
            for Array<T, ($($xyz,)+)>
        {
            #[inline]
            fn from(value: &$array) -> Self {
                Self::from_expr(View::from(value).cloned())
            }
        }

        impl<T $(,const $abc: usize)+> From<Array<T, ($(Const<$abc>,)+)>> for $array {
            #[inline]
            fn from(value: Array<T, ($(Const<$abc>,)+)>) -> Self {
                unsafe { mem::transmute_copy(&ManuallyDrop::new(value)) }
            }
        }

        impl<T $(,$xyz: Dim + From<Const<$abc>>)+ $(,const $abc: usize)+> From<$array>
            for Array<T, ($($xyz,)+)>
        {
            #[inline]
            fn from(value: $array) -> Self {
                let array: Array<T, ($(Const<$abc>,)+)> =
                    unsafe { mem::transmute_copy(&ManuallyDrop::new(value)) };

                array.into_buffer()
            }
        }
    };
}

impl_from_array!((X), (A), [T; A]);
impl_from_array!((X, Y), (A, B), [[T; B]; A]);
impl_from_array!((X, Y, Z), (A, B, C), [[[T; C]; B]; A]);
impl_from_array!((X, Y, Z, W), (A, B, C, D), [[[[T; D]; C]; B]; A]);
impl_from_array!((X, Y, Z, W, U), (A, B, C, D, E), [[[[[T; E]; D]; C]; B]; A]);
impl_from_array!((X, Y, Z, W, U, V), (A, B, C, D, E, F), [[[[[[T; F]; E]; D]; C]; B]; A]);

impl<T, S: Shape> FromExpression<T, S> for Array<T, S> {
    #[cfg(not(feature = "nightly"))]
    #[inline]
    fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self {
        Self::from_expr(expr.into_expr())
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self {
        Self::from_expr_in(expr.into_expr(), Global)
    }
}

impl<T> FromIterator<T> for Array<T, (Dyn,)> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Vec::from_iter(iter))
    }
}

impl<T: Hash, S: Shape, A: Allocator> Hash for Array<T, S, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T, S: Shape, A: Allocator, I: SliceIndex<T, S, Dense>> Index<I> for Array<T, S, A> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, S: Shape, A: Allocator, I: SliceIndex<T, S, Dense>> IndexMut<I> for Array<T, S, A> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoExpression for &'a Array<T, S, A> {
    type Shape = S;
    type IntoExpr = View<'a, T, S>;

    #[inline]
    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoExpression for &'a mut Array<T, S, A> {
    type Shape = S;
    type IntoExpr = ViewMut<'a, T, S>;

    #[inline]
    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<T, S: Shape, A: Allocator> IntoExpression for Array<T, S, A> {
    type Shape = S;
    type IntoExpr = IntoExpr<T, S::Buffer<ManuallyDrop<T>, A>>;

    #[inline]
    fn into_expr(self) -> Self::IntoExpr {
        unsafe { IntoExpr::new(self.buffer.cast()) }
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoIterator for &'a Array<T, S, A> {
    type Item = &'a T;
    type IntoIter = Iter<View<'a, T, S>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoIterator for &'a mut Array<T, S, A> {
    type Item = &'a mut T;
    type IntoIter = Iter<ViewMut<'a, T, S>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, S: Shape, A: Allocator> IntoIterator for Array<T, S, A> {
    type Item = T;
    type IntoIter = Iter<IntoExpr<T, S::Buffer<ManuallyDrop<T>, A>>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_expr().into_iter()
    }
}
