#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::collections::TryReserveError;
use std::mem::{self, MaybeUninit};
use std::ops::RangeBounds;
use std::{ptr, slice};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::GridArray;
use crate::buffer::GridBuffer;
use crate::dim::{Const, Dim, Shape};
use crate::expr::{self, Drain, IntoExpr};
use crate::expression::Expression;
use crate::mapping::{DenseMapping, Mapping};
use crate::traits::{Apply, IntoCloned, IntoExpression};

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

impl<T, D: Dim, A: Allocator> GridArray<T, D, A> {
    /// Returns a reference to the underlying allocator.
    #[cfg(feature = "nightly")]
    pub fn allocator(&self) -> &A {
        self.buffer.allocator()
    }

    /// Moves all elements from another array into the array along the outermost dimension.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match.
    pub fn append(&mut self, other: &mut Self) {
        self.expand(other.drain(..));
    }

    /// Returns the number of elements the array can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Clears the array, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity of the array.
    pub fn clear(&mut self) {
        unsafe {
            self.buffer.with_mut_vec(|vec| vec.clear());
            self.buffer.set_mapping(DenseMapping::default());
        }
    }

    /// Removes the specified range from the array along the outermost dimension,
    /// and returns the removed range as an expression.
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> Expression<Drain<T, D, A>> {
        #[cfg(not(feature = "nightly"))]
        let range = crate::index::range(range, ..self.size(D::RANK - 1));
        #[cfg(feature = "nightly")]
        let range = slice::range(range, ..self.size(D::RANK - 1));

        Expression::new(Drain::new(self, range.start, range.end))
    }

    /// Appends an expression to the array along the outermost dimension with broadcasting,
    /// cloning elements if needed.
    ///
    /// If the rank of the expression equals one less than the rank of the array,
    /// the expression is assumed to have outermost dimension of size 1.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match, or if the rank of the expression
    /// is not valid.
    pub fn expand<I: IntoExpression<Item: IntoCloned<T>>>(&mut self, expr: I) {
        assert!(I::Dim::RANK == D::RANK - 1 || I::Dim::RANK == D::RANK, "invalid rank");

        let expr = expr.into_expr();
        let len = expr.len();

        if len > 0 {
            let inner_shape = &expr.shape()[..D::RANK - 1];
            let mut shape = self.shape();

            if self.is_empty() {
                shape[..D::RANK - 1].copy_from_slice(inner_shape);
            } else {
                assert!(inner_shape == &shape[..D::RANK - 1], "inner dimensions mismatch");
            };

            shape[D::RANK - 1] += if I::Dim::RANK < D::RANK { 1 } else { expr.size(D::RANK - 1) };

            unsafe {
                self.buffer.with_mut_vec(|vec| {
                    vec.reserve(len);
                    expr.clone_into_vec(vec);
                });

                self.set_shape(shape);
            }
        }
    }

    /// Creates an array from the given element with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn from_elem_in(shape: D::Shape, elem: T, alloc: A) -> Self
    where
        T: Clone,
    {
        expr::from_elem(shape, elem).eval_in(alloc)
    }

    /// Creates an array from the given expression with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn from_expr_in<I: IntoExpression<Item = T, Dim = D>>(expr: I, alloc: A) -> Self {
        let expr = expr.into_expr();
        let shape = expr.shape();

        let mut vec = Vec::with_capacity_in(expr.len(), alloc);

        expr.clone_into_vec(&mut vec);

        unsafe { Self::from_parts(vec, DenseMapping::new(shape)) }
    }

    /// Creates an array with the results from the given function with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn from_fn_in<F: FnMut(D::Shape) -> T>(shape: D::Shape, f: F, alloc: A) -> Self {
        expr::from_fn(shape, f).eval_in(alloc)
    }

    /// Creates an array from raw components of another array with the specified allocator.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape, capacity and allocator.
    #[cfg(feature = "nightly")]
    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        shape: D::Shape,
        capacity: usize,
        alloc: A,
    ) -> Self {
        let mapping = DenseMapping::new(shape);

        Self::from_parts(Vec::from_raw_parts_in(ptr, mapping.len(), capacity, alloc), mapping)
    }

    /// Converts the array into a one-dimensional array.
    pub fn into_flattened(self) -> GridArray<T, Const<1>, A> {
        self.into_vec().into()
    }

    /// Decomposes an array into its raw components including the allocator.
    #[cfg(feature = "nightly")]
    pub fn into_raw_parts_with_alloc(self) -> (*mut T, D::Shape, usize, A) {
        let (vec, mapping) = self.buffer.into_parts();
        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_alloc();

        (ptr, mapping.shape(), capacity, alloc)
    }

    /// Converts an array with a single element into the contained value.
    ///
    /// # Panics
    ///
    /// Panics if the array length is not equal to one.
    pub fn into_scalar(self) -> T {
        assert!(self.len() == 1, "invalid length");

        self.into_vec().pop().unwrap()
    }

    /// Converts the array into a reshaped array, which must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed.
    pub fn into_shape<S: Shape>(self, shape: S) -> GridArray<T, S::Dim, A> {
        let (vec, mapping) = self.buffer.into_parts();

        unsafe { GridArray::from_parts(vec, Mapping::reshape(mapping, shape)) }
    }

    /// Converts the array into a vector.
    pub fn into_vec(self) -> vec_t!(T, A) {
        let (vec, _) = self.buffer.into_parts();

        vec
    }

    /// Creates a new, empty array with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn new_in(alloc: A) -> Self {
        unsafe { Self::from_parts(Vec::new_in(alloc), DenseMapping::default()) }
    }

    /// Reserves capacity for at least the additional number of elements in the array.
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            self.buffer.with_mut_vec(|vec| vec.reserve(additional));
        }
    }

    /// Reserves the minimum capacity for the additional number of elements in the array.
    pub fn reserve_exact(&mut self, additional: usize) {
        unsafe {
            self.buffer.with_mut_vec(|vec| vec.reserve_exact(additional));
        }
    }

    /// Resizes the array to the new shape, creating new elements with the given value.
    pub fn resize(&mut self, new_shape: D::Shape, value: T)
    where
        T: Clone,
        A: Clone,
    {
        self.buffer.resize_with(new_shape, || value.clone());
    }

    /// Resizes the array to the new shape, creating new elements from the given closure.
    pub fn resize_with<F: FnMut() -> T>(&mut self, new_shape: D::Shape, f: F)
    where
        A: Clone,
    {
        self.buffer.resize_with(new_shape, f);
    }

    /// Forces the array shape to the new shape.
    ///
    /// # Safety
    ///
    /// All elements within the array length must be initialized.
    pub unsafe fn set_shape(&mut self, new_shape: D::Shape) {
        self.buffer.set_mapping(DenseMapping::new(new_shape));
    }

    /// Shrinks the capacity of the array with a lower bound.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        unsafe {
            self.buffer.with_mut_vec(|vec| vec.shrink_to(min_capacity));
        }
    }

    /// Shrinks the capacity of the array as much as possible.
    pub fn shrink_to_fit(&mut self) {
        unsafe {
            self.buffer.with_mut_vec(|vec| vec.shrink_to_fit());
        }
    }

    /// Returns the remaining spare capacity of the array as a slice of `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the array with data, before marking
    /// the data as initialized using the `set_shape` method.
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let ptr = self.as_mut_ptr();
        let len = self.capacity() - self.len();

        unsafe { slice::from_raw_parts_mut(ptr.add(self.len()) as *mut MaybeUninit<T>, len) }
    }

    /// Shortens the array along the outermost dimension, keeping the first `size` elements.
    pub fn truncate(&mut self, size: usize) {
        if size < self.size(D::RANK - 1) {
            let new_mapping = self.mapping().resize_dim(D::RANK - 1, size);

            unsafe {
                self.buffer.with_mut_vec(|vec| vec.truncate(new_mapping.len()));
                self.buffer.set_mapping(new_mapping);
            }
        }
    }

    /// Tries to reserve capacity for at least the additional number of elements in the array.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.buffer.with_mut_vec(|vec| vec.try_reserve(additional)) }
    }

    /// Tries to reserve the minimum capacity for the additional number of elements in the array.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.buffer.with_mut_vec(|vec| vec.try_reserve_exact(additional)) }
    }

    /// Creates a new, empty array with the specified capacity and allocator.
    #[cfg(feature = "nightly")]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        unsafe { Self::from_parts(Vec::with_capacity_in(capacity, alloc), DenseMapping::default()) }
    }

    pub(crate) unsafe fn from_parts(vec: vec_t!(T, A), mapping: DenseMapping<D>) -> Self {
        Self { buffer: GridBuffer::from_parts(vec, mapping) }
    }

    pub(crate) fn zip_with<I: IntoExpression, F>(mut self, expr: I, mut f: F) -> Self
    where
        F: FnMut(T, I::Item) -> T,
    {
        struct DropGuard<'a, T, D: Dim, A: Allocator> {
            grid: &'a mut GridArray<T, D, A>,
            index: usize,
        }

        impl<'a, T, D: Dim, A: Allocator> Drop for DropGuard<'a, T, D, A> {
            fn drop(&mut self) {
                let ptr = self.grid.as_mut_ptr();
                let tail = self.grid.len() - self.index;

                // Drop all elements except the current one, which is read but not written back.
                unsafe {
                    self.grid.buffer.set_mapping(DenseMapping::default());

                    if self.index > 1 {
                        ptr::drop_in_place(ptr::slice_from_raw_parts_mut(ptr, self.index - 1));
                    }

                    ptr::drop_in_place(ptr::slice_from_raw_parts_mut(ptr.add(self.index), tail));
                }
            }
        }

        let mut guard = DropGuard { grid: &mut self, index: 0 };
        let expr = guard.grid.expr_mut().zip(expr);

        expr.for_each(|(x, y)| unsafe {
            guard.index += 1;
            ptr::write(x, f(ptr::read(x), y));
        });

        mem::forget(guard);

        self
    }
}

#[cfg(not(feature = "nightly"))]
impl<T, D: Dim> GridArray<T, D> {
    /// Creates an array from the given element.
    pub fn from_elem(shape: D::Shape, elem: T) -> Self
    where
        T: Clone,
    {
        expr::from_elem(shape, elem).eval()
    }

    /// Creates an array from the given expression.
    pub fn from_expr<I: IntoExpression<Item = T, Dim = D>>(expr: I) -> Self {
        let expr = expr.into_expr();
        let shape = expr.shape();

        let mut vec = Vec::with_capacity(expr.len());

        expr.clone_into_vec(&mut vec);

        unsafe { Self::from_parts(vec, DenseMapping::new(shape)) }
    }

    /// Creates an array with the results from the given function.
    pub fn from_fn<F: FnMut(D::Shape) -> T>(shape: D::Shape, f: F) -> Self {
        expr::from_fn(shape, f).eval()
    }

    /// Creates an array from raw components of another array.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape and capacity.
    pub unsafe fn from_raw_parts(ptr: *mut T, shape: D::Shape, capacity: usize) -> Self {
        let mapping = DenseMapping::new(shape);
        let vec = Vec::from_raw_parts(ptr, mapping.len(), capacity);

        Self::from_parts(vec, mapping)
    }

    /// Decomposes an array into its raw components.
    pub fn into_raw_parts(self) -> (*mut T, D::Shape, usize) {
        let (vec, mapping) = self.buffer.into_parts();
        let mut vec = mem::ManuallyDrop::new(vec);

        (vec.as_mut_ptr(), mapping.shape(), vec.capacity())
    }

    /// Creates a new, empty array.
    pub fn new() -> Self {
        unsafe { Self::from_parts(Vec::new(), DenseMapping::default()) }
    }

    /// Creates a new, empty array with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        unsafe { Self::from_parts(Vec::with_capacity(capacity), DenseMapping::default()) }
    }
}

#[cfg(feature = "nightly")]
impl<T, D: Dim> GridArray<T, D> {
    /// Creates an array from the given element.
    pub fn from_elem(shape: D::Shape, elem: T) -> Self
    where
        T: Clone,
    {
        Self::from_elem_in(shape, elem, Global)
    }

    /// Creates an array from the given expression.
    pub fn from_expr<I: IntoExpression<Item = T, Dim = D>>(expr: I) -> Self {
        Self::from_expr_in(expr, Global)
    }

    /// Creates an array with the results from the given function.
    pub fn from_fn<F: FnMut(D::Shape) -> T>(shape: D::Shape, f: F) -> Self {
        Self::from_fn_in(shape, f, Global)
    }

    /// Creates an array from raw components of another array.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape and capacity.
    pub unsafe fn from_raw_parts(ptr: *mut T, shape: D::Shape, capacity: usize) -> Self {
        Self::from_raw_parts_in(ptr, shape, capacity, Global)
    }

    /// Decomposes an array into its raw components.
    pub fn into_raw_parts(self) -> (*mut T, D::Shape, usize) {
        let (ptr, shape, capacity, _) = self.into_raw_parts_with_alloc();

        (ptr, shape, capacity)
    }

    /// Creates a new, empty array.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a new, empty array with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<T, D: Dim, A: Allocator> Apply<T> for GridArray<T, D, A> {
    type Output<F: FnMut(T) -> T> = Self;
    type ZippedWith<I: IntoExpression, F: FnMut(Self::Item, I::Item) -> T> = Self;

    fn apply<F: FnMut(T) -> T>(self, mut f: F) -> Self {
        self.zip_with(expr::fill(()), |x, ()| f(x))
    }

    fn zip_with<I: IntoExpression, F: FnMut(T, I::Item) -> T>(self, expr: I, f: F) -> Self {
        self.zip_with(expr, f)
    }
}

impl<T, D: Dim> Default for GridArray<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: 'a + Copy, A: Allocator> Extend<&'a T> for GridArray<T, Const<1>, A> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

impl<T, A: Allocator> Extend<T> for GridArray<T, Const<1>, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        unsafe {
            let len = self.buffer.with_mut_vec(|vec| {
                vec.extend(iter);
                vec.len()
            });

            self.set_shape([len]);
        }
    }
}

impl<T: Clone> From<&[T]> for GridArray<T, Const<1>> {
    fn from(slice: &[T]) -> Self {
        Self::from(slice.to_vec())
    }
}

macro_rules! impl_from_array {
    ($n:tt, ($($size:tt),+), $array:tt) => {
        impl<T, $(const $size: usize),+> From<$array> for GridArray<T, Const<$n,>> {
            #[cfg(not(feature = "nightly"))]
            fn from(array: $array) -> Self {
                if [$($size),+].contains(&0) {
                    Self::new()
                } else {
                    let mut vec = std::mem::ManuallyDrop::new(Vec::from(array));
                    let (ptr, capacity) = (vec.as_mut_ptr(), vec.capacity());

                    unsafe {
                        let capacity = capacity * (mem::size_of_val(&*ptr) / mem::size_of::<T>());

                        Self::from_raw_parts(ptr.cast(), [$($size),+], capacity)
                    }
                }
            }

            #[cfg(feature = "nightly")]
            fn from(array: $array) -> Self {
                if [$($size),+].contains(&0) {
                    Self::new()
                } else {
                    let (ptr, _, capacity, alloc) = Vec::from(array).into_raw_parts_with_alloc();

                    unsafe {
                        let capacity = capacity * (mem::size_of_val(&*ptr) / mem::size_of::<T>());

                        Self::from_raw_parts_in(ptr.cast(), [$($size),+], capacity, alloc)
                    }
                }
            }
        }
    };
}

impl_from_array!(1, (X), [T; X]);
impl_from_array!(2, (X, Y), [[T; X]; Y]);
impl_from_array!(3, (X, Y, Z), [[[T; X]; Y]; Z]);
impl_from_array!(4, (X, Y, Z, W), [[[[T; X]; Y]; Z]; W]);
impl_from_array!(5, (X, Y, Z, W, U), [[[[[T; X]; Y]; Z]; W]; U]);
impl_from_array!(6, (X, Y, Z, W, U, V), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

impl<T, D: Dim, A: Allocator> From<GridArray<T, D, A>> for vec_t!(T, A) {
    fn from(grid: GridArray<T, D, A>) -> Self {
        grid.into_vec()
    }
}

impl<T, A: Allocator> From<vec_t!(T, A)> for GridArray<T, Const<1>, A> {
    fn from(vec: vec_t!(T, A)) -> Self {
        let mapping = DenseMapping::new([vec.len()]);

        unsafe { Self::from_parts(vec, mapping) }
    }
}

impl<T> FromIterator<T> for GridArray<T, Const<1>> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Vec::from_iter(iter))
    }
}

impl<T, D: Dim, A: Allocator> IntoExpression for GridArray<T, D, A> {
    type Item = T;
    type Dim = D;
    type Producer = IntoExpr<T, D, A>;

    fn into_expr(self) -> Expression<Self::Producer> {
        Expression::new(IntoExpr::new(self))
    }
}

impl<T, D: Dim, A: Allocator> IntoIterator for GridArray<T, D, A> {
    type Item = T;
    type IntoIter = <vec_t!(T, A) as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}
