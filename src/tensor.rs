#[cfg(feature = "nightly")]
use alloc::alloc::{Allocator, Global};
#[cfg(not(feature = "std"))]
use alloc::borrow::ToOwned;
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
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
use crate::array::Array;
use crate::dim::{Const, Dim, Dyn};
use crate::expr::{self, Drain, IntoExpr, Iter, Map, Zip};
use crate::expr::{Apply, Expression, FromExpression, IntoExpression};
use crate::index::SliceIndex;
use crate::layout::{Dense, Layout};
use crate::mapping::{DenseMapping, Mapping};
use crate::raw_tensor::RawTensor;
use crate::shape::{ConstShape, DynRank, IntoShape, Rank, Shape};
use crate::slice::Slice;
use crate::traits::{IntoCloned, Owned};
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
pub struct Tensor<T, S: Shape = DynRank, A: Allocator = Global> {
    tensor: RawTensor<T, S, A>,
}

/// Multidimensional array with dynamically-sized dimensions and dense layout.
pub type DTensor<T, const N: usize, A = Global> = Tensor<T, Rank<N>, A>;

impl<T, S: Shape, A: Allocator> Tensor<T, S, A> {
    /// Returns a reference to the underlying allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn allocator(&self) -> &A {
        self.tensor.allocator()
    }

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
        self.tensor.capacity()
    }

    /// Clears the array, removing all values.
    ///
    /// If the array type has dynamic rank, the rank is set to 1.
    ///
    /// Note that this method has no effect on the allocated capacity of the array.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[inline]
    pub fn clear(&mut self) {
        assert!(S::default().len() == 0, "default length not zero");

        unsafe {
            self.tensor.with_mut_parts(|vec, mapping| {
                vec.clear();
                *mapping = DenseMapping::default();
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
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> IntoExpr<Drain<'_, T, S, A>> {
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
                self.tensor.with_mut_parts(|vec, mapping| {
                    vec.reserve(len);

                    expr.shape().with_dims(|src| {
                        if mapping.is_empty() {
                            if src.len() == mapping.rank() {
                                mapping.shape_mut().with_mut_dims(|dims| dims.copy_from_slice(src));
                            } else {
                                *mapping = DenseMapping::new(Shape::from_dims(src));
                            }
                        } else {
                            mapping.shape_mut().with_mut_dims(|dims| {
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

    /// Creates an array from the given element with the specified allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn from_elem_in<I: IntoShape<IntoShape = S>>(shape: I, elem: T, alloc: A) -> Self
    where
        T: Clone,
    {
        Self::from_expr_in(expr::from_elem(shape, elem), alloc)
    }

    /// Creates an array with the results from the given function with the specified allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn from_fn_in<I: IntoShape<IntoShape = S>, F>(shape: I, f: F, alloc: A) -> Self
    where
        F: FnMut(&[usize]) -> T,
    {
        Self::from_expr_in(expr::from_fn(shape, f), alloc)
    }

    /// Creates an array from raw components of another array with the specified allocator.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the mapping, capacity and allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        mapping: DenseMapping<S>,
        capacity: usize,
        alloc: A,
    ) -> Self {
        unsafe {
            Self::from_parts(Vec::from_raw_parts_in(ptr, mapping.len(), capacity, alloc), mapping)
        }
    }

    /// Converts the array into an array with dynamic rank.
    #[inline]
    pub fn into_dyn(self) -> Tensor<T, DynRank, A> {
        self.into_mapping()
    }

    /// Converts the array into a one-dimensional array.
    #[inline]
    pub fn into_flat(self) -> Tensor<T, (Dyn,), A> {
        self.into_vec().into()
    }

    /// Converts the array into a remapped array.
    ///
    /// # Panics
    ///
    /// Panics if the shape is not matching static rank or constant-sized dimensions.
    #[inline]
    pub fn into_mapping<R: Shape>(self) -> Tensor<T, R, A> {
        let (vec, mapping) = self.tensor.into_parts();

        unsafe { Tensor::from_parts(vec, Mapping::remap(&mapping)) }
    }

    /// Decomposes an array into its raw components including the allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn into_raw_parts_with_alloc(self) -> (*mut T, DenseMapping<S>, usize, A) {
        let (vec, mapping) = self.tensor.into_parts();
        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_alloc();

        (ptr, mapping, capacity, alloc)
    }

    /// Converts an array with a single element into the contained value.
    ///
    /// # Panics
    ///
    /// Panics if the array length is not equal to one.
    #[inline]
    pub fn into_scalar(self) -> T {
        assert!(self.len() == 1, "invalid length");

        self.into_vec().pop().unwrap()
    }

    /// Converts the array into a reshaped array, which must have the same length.
    ///
    /// At most one dimension can have dynamic size `usize::MAX`, and is then inferred
    /// from the other dimensions and the array length.
    ///
    /// # Examples
    ///
    /// ```
    /// use mdarray::{tensor, view};
    ///
    /// let t = tensor![[1, 2, 3], [4, 5, 6]];
    ///
    /// assert_eq!(t.into_shape([!0, 2]), view![[1, 2], [3, 4], [5, 6]]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed.
    #[inline]
    pub fn into_shape<I: IntoShape>(self, shape: I) -> Tensor<T, I::IntoShape, A> {
        let (vec, mapping) = self.tensor.into_parts();

        unsafe { Tensor::from_parts(vec, mapping.reshape(shape.into_shape())) }
    }

    /// Converts the array into a vector.
    #[inline]
    pub fn into_vec(self) -> vec_t!(T, A) {
        let (vec, _) = self.tensor.into_parts();

        vec
    }

    /// Returns the array with the given closure applied to each element.
    #[inline]
    pub fn map<F: FnMut(T) -> T>(self, mut f: F) -> Self {
        self.zip_with(expr::fill(()), |(x, ())| f(x))
    }

    /// Creates a new, empty array with the specified allocator.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn new_in(alloc: A) -> Self {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe { Self::from_parts(Vec::new_in(alloc), DenseMapping::default()) }
    }

    /// Reserves capacity for at least the additional number of elements in the array.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            self.tensor.with_mut_parts(|vec, _| vec.reserve(additional));
        }
    }

    /// Reserves the minimum capacity for the additional number of elements in the array.
    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        unsafe {
            self.tensor.with_mut_parts(|vec, _| vec.reserve_exact(additional));
        }
    }

    /// Resizes the array to the new shape, creating new elements with the given value.
    #[inline]
    pub fn resize(&mut self, new_dims: &[usize], value: T)
    where
        T: Clone,
        A: Clone,
    {
        self.tensor.resize_with(new_dims, || value.clone());
    }

    /// Resizes the array to the new shape, creating new elements from the given closure.
    #[inline]
    pub fn resize_with<F: FnMut() -> T>(&mut self, new_dims: &[usize], f: F)
    where
        A: Clone,
    {
        self.tensor.resize_with(new_dims, f);
    }

    /// Forces the array layout mapping to the new mapping.
    ///
    /// # Safety
    ///
    /// All elements within the array length must be initialized.
    #[inline]
    pub unsafe fn set_mapping(&mut self, new_mapping: DenseMapping<S>) {
        unsafe {
            self.tensor.set_mapping(new_mapping);
        }
    }

    /// Shrinks the capacity of the array with a lower bound.
    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        unsafe {
            self.tensor.with_mut_parts(|vec, _| vec.shrink_to(min_capacity));
        }
    }

    /// Shrinks the capacity of the array as much as possible.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        unsafe {
            self.tensor.with_mut_parts(|vec, _| vec.shrink_to_fit());
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
                self.tensor.with_mut_parts(|vec, mapping| {
                    mapping.shape_mut().with_mut_dims(|dims| dims[0] = size);
                    vec.truncate(mapping.len());
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
        unsafe { self.tensor.with_mut_parts(|vec, _| vec.try_reserve(additional)) }
    }

    /// Tries to reserve the minimum capacity for the additional number of elements in the array.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.tensor.with_mut_parts(|vec, _| vec.try_reserve_exact(additional)) }
    }

    /// Creates an array with uninitialized elements and the specified allocator.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn uninit_in<I: IntoShape<IntoShape = S>>(
        shape: I,
        alloc: A,
    ) -> Tensor<MaybeUninit<T>, S, A> {
        let shape = shape.into_shape();
        let len = shape.checked_len().expect("invalid length");

        let vec = Vec::from(Box::new_uninit_slice_in(len, alloc));

        unsafe { Tensor::from_parts(vec, DenseMapping::new(shape)) }
    }

    /// Creates a new, empty array with the specified capacity and allocator.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe { Self::from_parts(Vec::with_capacity_in(capacity, alloc), DenseMapping::default()) }
    }

    /// Creates an array with elements set to zero.
    ///
    /// Zero elements are created using `Default::default()`.
    #[cfg(feature = "nightly")]
    #[inline]
    pub fn zeros_in<I: IntoShape<IntoShape = S>>(shape: I, alloc: A) -> Self
    where
        T: Default,
    {
        let mut tensor = Tensor::uninit_in(shape, alloc);

        tensor.expr_mut().for_each(|x| {
            _ = x.write(T::default());
        });

        unsafe { tensor.assume_init() }
    }

    #[cfg(not(feature = "nightly"))]
    #[inline]
    fn from_expr<E: Expression<Item = T, Shape = S>>(expr: E) -> Self {
        let shape = expr.shape().clone();
        let mut vec = Vec::with_capacity(shape.len());

        expr.clone_into_vec(&mut vec);

        unsafe { Self::from_parts(vec, DenseMapping::new(shape)) }
    }

    #[cfg(feature = "nightly")]
    #[inline]
    pub(crate) fn from_expr_in<E>(expr: E, alloc: A) -> Self
    where
        E: Expression<Item = T, Shape = S>,
    {
        let shape = expr.shape().clone();
        let mut vec = Vec::with_capacity_in(shape.len(), alloc);

        expr.clone_into_vec(&mut vec);

        unsafe { Self::from_parts(vec, DenseMapping::new(shape)) }
    }

    #[inline]
    pub(crate) unsafe fn from_parts(vec: vec_t!(T, A), mapping: DenseMapping<S>) -> Self {
        unsafe { Self { tensor: RawTensor::from_parts(vec, mapping) } }
    }

    #[inline]
    fn zip_with<I: IntoExpression, F>(self, expr: I, mut f: F) -> Self
    where
        F: FnMut((T, I::Item)) -> T,
    {
        struct DropGuard<T, S: Shape, A: Allocator> {
            tensor: ManuallyDrop<Tensor<T, S, A>>,
            index: usize,
        }

        impl<T, S: Shape, A: Allocator> Drop for DropGuard<T, S, A> {
            #[inline]
            fn drop(&mut self) {
                let ptr = self.tensor.as_mut_ptr();
                let tail = self.tensor.len() - self.index;

                // Drop all elements except the current one, which is read but not written back.
                unsafe {
                    let mut vec = ManuallyDrop::take(&mut self.tensor).into_vec();

                    vec.set_len(0);

                    if self.index > 1 {
                        ptr::slice_from_raw_parts_mut(ptr, self.index - 1).drop_in_place();
                    }

                    ptr::slice_from_raw_parts_mut(ptr.add(self.index), tail).drop_in_place();
                }
            }
        }

        let mut guard = DropGuard { tensor: ManuallyDrop::new(self), index: 0 };
        let expr = guard.tensor.expr_mut().zip(expr);

        expr.for_each(|(x, y)| unsafe {
            guard.index += 1;
            ptr::write(x, f((ptr::read(x), y)));
        });

        let tensor = unsafe { ManuallyDrop::take(&mut guard.tensor) };

        mem::forget(guard);

        tensor
    }
}

#[cfg(not(feature = "nightly"))]
impl<T, S: Shape> Tensor<T, S> {
    /// Creates an array from the given element.
    #[inline]
    pub fn from_elem<I: IntoShape<IntoShape = S>>(shape: I, elem: T) -> Self
    where
        T: Clone,
    {
        Self::from_expr(expr::from_elem(shape, elem))
    }

    /// Creates an array with the results from the given function.
    #[inline]
    pub fn from_fn<I: IntoShape<IntoShape = S>, F>(shape: I, f: F) -> Self
    where
        F: FnMut(&[usize]) -> T,
    {
        Self::from_expr(expr::from_fn(shape, f))
    }

    /// Creates an array from raw components of another array.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape and capacity.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, mapping: DenseMapping<S>, capacity: usize) -> Self {
        unsafe { Self::from_parts(Vec::from_raw_parts(ptr, mapping.len(), capacity), mapping) }
    }

    /// Decomposes an array into its raw components.
    #[inline]
    pub fn into_raw_parts(self) -> (*mut T, DenseMapping<S>, usize) {
        let (vec, mapping) = self.tensor.into_parts();
        let mut vec = mem::ManuallyDrop::new(vec);

        (vec.as_mut_ptr(), mapping, vec.capacity())
    }

    /// Creates a new, empty array.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[inline]
    pub fn new() -> Self {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe { Self::from_parts(Vec::new(), DenseMapping::default()) }
    }

    /// Creates an array with uninitialized elements.
    #[inline]
    pub fn uninit<I: IntoShape<IntoShape = S>>(shape: I) -> Tensor<MaybeUninit<T>, S> {
        let shape = shape.into_shape();
        let len = shape.checked_len().expect("invalid length");

        let vec = Vec::from(Box::new_uninit_slice(len));

        unsafe { Tensor::from_parts(vec, DenseMapping::new(shape)) }
    }

    /// Creates a new, empty array with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe { Self::from_parts(Vec::with_capacity(capacity), DenseMapping::default()) }
    }

    /// Creates an array with elements set to zero.
    ///
    /// Zero elements are created using `Default::default()`.
    #[inline]
    pub fn zeros<I: IntoShape<IntoShape = S>>(shape: I) -> Self
    where
        T: Default,
    {
        let mut tensor = Tensor::uninit(shape);

        tensor.expr_mut().for_each(|x| {
            _ = x.write(T::default());
        });

        unsafe { tensor.assume_init() }
    }
}

#[cfg(feature = "nightly")]
impl<T, S: Shape> Tensor<T, S> {
    /// Creates an array from the given element.
    #[inline]
    pub fn from_elem<I: IntoShape<IntoShape = S>>(shape: I, elem: T) -> Self
    where
        T: Clone,
    {
        Self::from_elem_in(shape, elem, Global)
    }

    /// Creates an array with the results from the given function.
    #[inline]
    pub fn from_fn<I: IntoShape<IntoShape = S>, F>(shape: I, f: F) -> Self
    where
        F: FnMut(&[usize]) -> T,
    {
        Self::from_fn_in(shape, f, Global)
    }

    /// Creates an array from raw components of another array.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape and capacity.
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, mapping: DenseMapping<S>, capacity: usize) -> Self {
        unsafe { Self::from_raw_parts_in(ptr, mapping, capacity, Global) }
    }

    /// Decomposes an array into its raw components.
    #[inline]
    pub fn into_raw_parts(self) -> (*mut T, DenseMapping<S>, usize) {
        let (ptr, mapping, capacity, _) = self.into_raw_parts_with_alloc();

        (ptr, mapping, capacity)
    }

    /// Creates a new, empty array.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[inline]
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates an array with uninitialized elements.
    #[inline]
    pub fn uninit<I: IntoShape<IntoShape = S>>(shape: I) -> Tensor<MaybeUninit<T>, S> {
        Self::uninit_in(shape, Global)
    }

    /// Creates a new, empty array with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
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
}

#[cfg(not(feature = "nightly"))]
impl<T, S: Shape, A: Allocator> Tensor<MaybeUninit<T>, S, A> {
    /// Converts the array element type from `MaybeUninit<T>` to `T`.
    ///
    /// # Safety
    ///
    /// All elements in the array must be initialized, or the behavior is undefined.
    #[inline]
    pub unsafe fn assume_init(self) -> Tensor<T, S, A> {
        let (vec, mapping) = self.tensor.into_parts();

        let mut vec = mem::ManuallyDrop::new(vec);
        let (ptr, len, capacity) = (vec.as_mut_ptr(), vec.len(), vec.capacity());

        unsafe { Tensor::from_parts(Vec::from_raw_parts(ptr.cast(), len, capacity), mapping) }
    }
}

#[cfg(feature = "nightly")]
impl<T, S: Shape, A: Allocator> Tensor<MaybeUninit<T>, S, A> {
    /// Converts the array element type from `MaybeUninit<T>` to `T`.
    ///
    /// # Safety
    ///
    /// All elements in the array must be initialized, or the behavior is undefined.
    #[inline]
    pub unsafe fn assume_init(self) -> Tensor<T, S, A> {
        let (ptr, mapping, capacity, alloc) = self.into_raw_parts_with_alloc();

        unsafe { Tensor::from_raw_parts_in(ptr.cast(), mapping, capacity, alloc) }
    }
}

impl<'a, T, U, S: Shape, A: Allocator> Apply<U> for &'a Tensor<T, S, A> {
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

impl<'a, T, U, S: Shape, A: Allocator> Apply<U> for &'a mut Tensor<T, S, A> {
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

impl<T, S: Shape, A: Allocator> Apply<T> for Tensor<T, S, A> {
    type Output<F: FnMut(T) -> T> = Self;
    type ZippedWith<I: IntoExpression, F: FnMut((T, I::Item)) -> T> = Self;

    #[inline]
    fn apply<F: FnMut(T) -> T>(self, f: F) -> Self {
        self.map(f)
    }

    #[inline]
    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self
    where
        F: FnMut((T, I::Item)) -> T,
    {
        self.zip_with(expr, f)
    }
}

impl<T, U: ?Sized, S: Shape, A: Allocator> AsMut<U> for Tensor<T, S, A>
where
    Slice<T, S>: AsMut<U>,
{
    #[inline]
    fn as_mut(&mut self) -> &mut U {
        (**self).as_mut()
    }
}

impl<T, U: ?Sized, S: Shape, A: Allocator> AsRef<U> for Tensor<T, S, A>
where
    Slice<T, S>: AsRef<U>,
{
    #[inline]
    fn as_ref(&self) -> &U {
        (**self).as_ref()
    }
}

impl<T, S: Shape, A: Allocator> Borrow<Slice<T, S>> for Tensor<T, S, A> {
    #[inline]
    fn borrow(&self) -> &Slice<T, S> {
        self
    }
}

impl<T, S: Shape, A: Allocator> BorrowMut<Slice<T, S>> for Tensor<T, S, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut Slice<T, S> {
        self
    }
}

impl<T: Clone, S: Shape, A: Allocator + Clone> Clone for Tensor<T, S, A> {
    #[inline]
    fn clone(&self) -> Self {
        Self { tensor: self.tensor.clone() }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.tensor.clone_from(&source.tensor);
    }
}

impl<T: Debug, S: Shape, A: Allocator> Debug for Tensor<T, S, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T: Default, S: Shape> Default for Tensor<T, S> {
    #[inline]
    fn default() -> Self {
        Self::zeros(S::default())
    }
}

impl<T, S: Shape, A: Allocator> Deref for Tensor<T, S, A> {
    type Target = Slice<T, S>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.tensor.as_slice()
    }
}

impl<T, S: Shape, A: Allocator> DerefMut for Tensor<T, S, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor.as_mut_slice()
    }
}

impl<'a, T: Copy, A: Allocator> Extend<&'a T> for Tensor<T, (Dyn,), A> {
    #[inline]
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

impl<T, A: Allocator> Extend<T> for Tensor<T, (Dyn,), A> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        unsafe {
            self.tensor.with_mut_parts(|vec, mapping| {
                vec.extend(iter);
                *mapping = DenseMapping::new((vec.len(),));
            });
        }
    }
}

impl<T: Clone> From<&[T]> for Tensor<T, (Dyn,)> {
    #[inline]
    fn from(value: &[T]) -> Self {
        Self::from(value.to_vec())
    }
}

impl<T, S: ConstShape> From<Array<T, S>> for Tensor<T, S> {
    #[inline]
    fn from(value: Array<T, S>) -> Self {
        Self::from_expr(value.into_expr())
    }
}

impl<'a, T: 'a + Clone, S: Shape, L: Layout, I: IntoExpression<IntoExpr = View<'a, T, S, L>>>
    From<I> for Tensor<T, S>
{
    #[inline]
    fn from(value: I) -> Self {
        Self::from_expr(value.into_expr().cloned())
    }
}

impl<T, D: Dim, A: Allocator> From<Tensor<T, (D,), A>> for vec_t!(T, A) {
    #[inline]
    fn from(value: Tensor<T, (D,), A>) -> Self {
        value.into_vec()
    }
}

impl<T, A: Allocator> From<vec_t!(T, A)> for Tensor<T, (Dyn,), A> {
    #[inline]
    fn from(value: vec_t!(T, A)) -> Self {
        let mapping = DenseMapping::new((value.len(),));

        unsafe { Self::from_parts(value, mapping) }
    }
}

macro_rules! impl_from_array {
    (($($xyz:tt),+), ($($abc:tt),+), $array:tt) => {
        impl<T: Clone $(,$xyz: Dim + From<Const<$abc>>)+ $(,const $abc: usize)+> From<&$array>
            for Tensor<T, ($($xyz,)+)>
        {
            #[inline]
            fn from(value: &$array) -> Self {
                Self::from_expr(View::from(value).cloned())
            }
        }

        impl<T $(,$xyz: Dim + From<Const<$abc>>)+ $(,const $abc: usize)+> From<$array>
            for Tensor<T, ($($xyz,)+)>
        {
            #[inline]
            fn from(value: $array) -> Self {
                let mapping = DenseMapping::new(($($xyz::from(Const::<$abc>),)+));
                let capacity = mapping.shape().checked_len().expect("invalid length");

                let ptr = Box::into_raw(Box::new(value));

                unsafe { Self::from_raw_parts(ptr.cast(), mapping, capacity) }
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

impl<T, S: Shape> FromExpression<T, S> for Tensor<T, S> {
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

impl<T> FromIterator<T> for Tensor<T, (Dyn,)> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Vec::from_iter(iter))
    }
}

impl<T: Hash, S: Shape, A: Allocator> Hash for Tensor<T, S, A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T, S: Shape, A: Allocator, I: SliceIndex<T, S, Dense>> Index<I> for Tensor<T, S, A> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, S: Shape, A: Allocator, I: SliceIndex<T, S, Dense>> IndexMut<I> for Tensor<T, S, A> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoExpression for &'a Tensor<T, S, A> {
    type Shape = S;
    type IntoExpr = View<'a, T, S>;

    #[inline]
    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoExpression for &'a mut Tensor<T, S, A> {
    type Shape = S;
    type IntoExpr = ViewMut<'a, T, S>;

    #[inline]
    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<T, S: Shape, A: Allocator> IntoExpression for Tensor<T, S, A> {
    type Shape = S;
    type IntoExpr = IntoExpr<Tensor<ManuallyDrop<T>, S, A>>;

    #[cfg(not(feature = "nightly"))]
    #[inline]
    fn into_expr(self) -> Self::IntoExpr {
        let (vec, mapping) = self.tensor.into_parts();

        let mut vec = mem::ManuallyDrop::new(vec);
        let (ptr, len, capacity) = (vec.as_mut_ptr(), vec.len(), vec.capacity());

        let tensor =
            unsafe { Tensor::from_parts(Vec::from_raw_parts(ptr.cast(), len, capacity), mapping) };

        IntoExpr::new(tensor)
    }

    #[cfg(feature = "nightly")]
    #[inline]
    fn into_expr(self) -> Self::IntoExpr {
        let (ptr, mapping, capacity, alloc) = self.into_raw_parts_with_alloc();

        let tensor = unsafe { Tensor::from_raw_parts_in(ptr.cast(), mapping, capacity, alloc) };

        IntoExpr::new(tensor)
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoIterator for &'a Tensor<T, S, A> {
    type Item = &'a T;
    type IntoIter = Iter<View<'a, T, S>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoIterator for &'a mut Tensor<T, S, A> {
    type Item = &'a mut T;
    type IntoIter = Iter<ViewMut<'a, T, S>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, S: Shape, A: Allocator> IntoIterator for Tensor<T, S, A> {
    type Item = T;
    type IntoIter = Iter<IntoExpr<Tensor<ManuallyDrop<T>, S, A>>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.into_expr().into_iter()
    }
}

impl<T, S: Shape> Owned<T, S> for Tensor<T, S> {
    type WithConst<const N: usize> = Tensor<T, S::Prepend<Const<N>>>;

    #[inline]
    fn clone_from_slice(&mut self, slice: &Slice<T, S>)
    where
        T: Clone,
    {
        unsafe {
            self.tensor.with_mut_parts(|vec, mapping| {
                slice[..].clone_into(vec);
                mapping.clone_from(slice.mapping());
            });
        }
    }
}

macro_rules! impl_try_from_array {
    (($($xyz:tt),+), ($($abc:tt),+), $array:tt) => {
        impl<T $(,$xyz: Dim)+ $(,const $abc: usize)+> TryFrom<Tensor<T, ($($xyz,)+)>> for $array {
            type Error = Tensor<T, ($($xyz,)+)>;

            #[inline]
            fn try_from(value: Tensor<T, ($($xyz,)+)>) -> Result<Self, Self::Error> {
                if value.shape().with_dims(|dims| dims == &[$($abc),+]) {
                    let mut vec = value.into_vec();

                    unsafe {
                        vec.set_len(0);

                        Ok((vec.as_ptr() as *const $array).read())
                    }
                } else {
                    Err(value)
                }
            }
        }
    };
}

impl_try_from_array!((X), (A), [T; A]);
impl_try_from_array!((X, Y), (A, B), [[T; B]; A]);
impl_try_from_array!((X, Y, Z), (A, B, C), [[[T; C]; B]; A]);
impl_try_from_array!((X, Y, Z, W), (A, B, C, D), [[[[T; D]; C]; B]; A]);
impl_try_from_array!((X, Y, Z, W, U), (A, B, C, D, E), [[[[[T; E]; D]; C]; B]; A]);
impl_try_from_array!((X, Y, Z, W, U, V), (A, B, C, D, E, F), [[[[[[T; F]; E]; D]; C]; B]; A]);
