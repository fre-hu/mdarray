#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::borrow::{Borrow, BorrowMut};
use std::collections::TryReserveError;
use std::fmt::{self, Debug, Formatter};
use std::iter::FromIterator;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::result::Result;

#[cfg(not(feature = "nightly"))]
use crate::alloc::{Allocator, Global};
use crate::buffer::{Buffer, BufferMut, DenseBuffer, SubBuffer, SubBufferMut};
use crate::dim::{Dim, Rank, Shape};
use crate::format::Format;
use crate::index::{Axis, Const, ViewIndex};
use crate::layout::{panic_bounds_check, DenseLayout, Layout, ValidLayout, ViewLayout};
use crate::order::Order;
use crate::span::{DenseSpan, SpanBase};

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

/// Multidimensional array with static rank and element order.
pub struct GridBase<B: Buffer> {
    buffer: B,
}

/// Dense multidimensional array with static rank and element order.
pub type DenseGrid<T, D, A = Global> = GridBase<DenseBuffer<T, D, A>>;

/// Multidimensional array view with static rank and element order.
pub type SubGrid<'a, T, L> = GridBase<SubBuffer<'a, T, L>>;

/// Mutable multidimensional array view with static rank and element order.
pub type SubGridMut<'a, T, L> = GridBase<SubBufferMut<'a, T, L>>;

impl<B: Buffer> GridBase<B> {
    /// Returns an array span of the entire array.
    pub fn as_span(&self) -> &SpanBase<B::Item, B::Layout> {
        unsafe { &*SpanBase::from_raw_parts(self.buffer.as_ptr(), self.buffer.layout()) }
    }
}

impl<B: BufferMut> GridBase<B> {
    /// Returns a mutable array span of the entire array.
    pub fn as_mut_span(&mut self) -> &mut SpanBase<B::Item, B::Layout> {
        unsafe {
            &mut *SpanBase::from_raw_parts_mut(self.buffer.as_mut_ptr(), self.buffer.layout())
        }
    }
}

impl<T, D: Dim, A: Allocator> DenseGrid<T, D, A> {
    /// Returns a reference to the underlying allocator.
    #[cfg(feature = "nightly")]
    pub fn allocator(&self) -> &A {
        self.buffer.vec().allocator()
    }

    /// Moves all elements from a source array into the array along the outer dimension.
    /// # Panics
    /// Panics if the inner dimensions do not match.
    pub fn append(&mut self, grid: &mut Self) {
        let shape = if self.buffer.vec().is_empty() {
            grid.shape()
        } else {
            let mut shape = self.shape();

            let dim = D::dim(D::RANK - 1);
            let inner_dims = D::dims(..D::RANK - 1);

            assert!(
                grid.shape()[inner_dims.clone()] == shape[inner_dims],
                "inner dimensions mismatch"
            );

            shape[dim] += grid.size(dim);
            shape
        };

        unsafe {
            grid.buffer.set_layout(Layout::default());
            self.buffer.vec_mut().append(grid.buffer.vec_mut());
            self.buffer.set_layout(DenseLayout::new(shape));
        }
    }

    /// Returns the number of elements the array can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.buffer.vec().capacity()
    }

    /// Clears the array, removing all values.
    pub fn clear(&mut self) {
        unsafe {
            self.buffer.vec_mut().clear();
            self.buffer.set_layout(Layout::default());
        }
    }

    /// Clones the array span and appends to the array along the outer dimension.
    /// # Panics
    /// Panics if the inner dimensions do not match.
    pub fn extend_from_span(&mut self, span: &SpanBase<T, Layout<D, impl Format>>)
    where
        T: Clone,
    {
        let shape = if self.buffer.vec().is_empty() {
            span.shape()
        } else {
            let mut shape = self.shape();

            let dim = D::dim(D::RANK - 1);
            let inner_dims = D::dims(..D::RANK - 1);

            assert!(
                span.shape()[inner_dims.clone()] == shape[inner_dims],
                "inner dimensions mismatch"
            );

            shape[dim] += span.size(dim);
            shape
        };

        self.reserve(span.len());

        unsafe {
            #[cfg(not(feature = "nightly"))]
            extend_from_span::<_, _, A>(self.buffer.vec_mut(), span);
            #[cfg(feature = "nightly")]
            extend_from_span(self.buffer.vec_mut(), span);

            self.buffer.set_layout(DenseLayout::new(shape));
        }
    }

    /// Creates an array with the results from the given function with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn from_fn_in<F: FnMut(D::Shape) -> T>(shape: D::Shape, mut f: F, alloc: A) -> Self {
        let len = shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));
        let mut vec = Vec::with_capacity_in(len, alloc);

        unsafe {
            from_fn::<T, D, A, D::Lower, F>(&mut vec, shape, D::Shape::default(), &mut f);

            Self::from_parts(vec, DenseLayout::new(shape))
        }
    }

    /// Creates an array from a vector and layout.
    /// # Safety
    /// The vector length must match the given layout.
    pub unsafe fn from_parts(vec: vec_t!(T, A), layout: DenseLayout<D>) -> Self {
        Self { buffer: DenseBuffer::from_parts(vec, layout) }
    }

    /// Converts the array into a one-dimensional array.
    pub fn into_flattened(self) -> DenseGrid<T, Rank<1, D::Order>, A> {
        let layout = DenseLayout::new([self.len()]);

        unsafe { DenseGrid::from_parts(self.into_vec(), layout) }
    }

    /// Decomposes an array into a vector and layout.
    pub fn into_parts(self) -> (vec_t!(T, A), DenseLayout<D>) {
        self.buffer.into_parts()
    }

    /// Converts the array into a reshaped array, which must have the same length.
    /// # Panics
    /// Panics if the array length is changed.
    pub fn into_shape<S: Shape>(self, shape: S) -> DenseGrid<T, S::Dim<D::Order>, A> {
        let layout = self.layout().reshape(shape);

        unsafe { DenseGrid::from_parts(self.into_vec(), layout) }
    }

    /// Converts the array into a vector.
    pub fn into_vec(self) -> vec_t!(T, A) {
        let (vec, _) = self.into_parts();

        vec
    }

    /// Returns the array with the given closure applied to each element.
    pub fn map(mut self, mut f: impl FnMut(T) -> T) -> Self
    where
        T: Default,
    {
        map(&mut self, &mut f);

        self
    }

    /// Creates a new, empty array with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn new_in(alloc: A) -> Self {
        unsafe { Self::from_parts(Vec::new_in(alloc), Layout::default()) }
    }

    /// Reserves capacity for at least the additional number of elements in the array.
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            self.buffer.vec_mut().reserve(additional);
        }
    }

    /// Reserves the minimum capacity for the additional number of elements in the array.
    pub fn reserve_exact(&mut self, additional: usize) {
        unsafe {
            self.buffer.vec_mut().reserve_exact(additional);
        }
    }

    /// Resizes the array to the given shape, creating new elements with the given value.
    pub fn resize(&mut self, shape: D::Shape, value: impl Borrow<T> + Copy)
    where
        T: Clone,
        A: Clone,
    {
        self.buffer.resize_with(shape, || value.borrow().clone());
    }

    /// Resizes the array to the given shape, creating new elements from the given closure.
    pub fn resize_with(&mut self, shape: D::Shape, f: impl FnMut() -> T)
    where
        T: Clone,
        A: Clone,
    {
        self.buffer.resize_with(shape, f);
    }

    /// Forces the array layout to the specified layout.
    /// # Safety
    /// All elements within the array length must be initialized.
    pub unsafe fn set_layout(&mut self, layout: DenseLayout<D>) {
        self.buffer.vec_mut().set_len(layout.len());
        self.buffer.set_layout(layout);
    }

    /// Shrinks the capacity of the array with a lower bound.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        unsafe {
            self.buffer.vec_mut().shrink_to(min_capacity);
        }
    }

    /// Shrinks the capacity of the array as much as possible.
    pub fn shrink_to_fit(&mut self) {
        unsafe {
            self.buffer.vec_mut().shrink_to_fit();
        }
    }

    /// Shortens the array along the outer dimension, keeping the first `size` elements.
    pub fn truncate(&mut self, size: usize) {
        let dim = D::dim(D::RANK - 1);

        if size < self.size(dim) {
            let len = size * self.stride(dim) as usize;

            unsafe {
                self.buffer.vec_mut().truncate(len);
                self.buffer.set_layout(self.layout().resize_dim(dim, size));
            }
        }
    }

    /// Tries to reserve capacity for at least the additional number of elements in the array.
    /// # Errors
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.buffer.vec_mut().try_reserve(additional) }
    }

    /// Tries to reserve the minimum capacity for the additional number of elements in the array.
    /// # Errors
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.buffer.vec_mut().try_reserve_exact(additional) }
    }

    /// Creates a new, empty array with the specified capacity and allocator.
    #[cfg(feature = "nightly")]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        unsafe { Self::from_parts(Vec::with_capacity_in(capacity, alloc), Layout::default()) }
    }
}

#[cfg(not(feature = "nightly"))]
impl<T, D: Dim> DenseGrid<T, D, Global> {
    /// Creates an array with the results from the given function.
    pub fn from_fn<F: FnMut(D::Shape) -> T>(shape: D::Shape, mut f: F) -> Self {
        let len = shape[..].iter().fold(1usize, |acc, &x| acc.saturating_mul(x));
        let mut vec = Vec::with_capacity(len);

        unsafe {
            from_fn::<T, D, Global, D::Lower, F>(&mut vec, shape, D::Shape::default(), &mut f);

            Self::from_parts(vec, DenseLayout::new(shape))
        }
    }

    /// Creates a new, empty array.
    pub fn new() -> Self {
        unsafe { Self::from_parts(Vec::new(), Layout::default()) }
    }

    /// Creates a new, empty array with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        unsafe { Self::from_parts(Vec::with_capacity(capacity), Layout::default()) }
    }
}

#[cfg(feature = "nightly")]
impl<T, D: Dim> DenseGrid<T, D, Global> {
    /// Creates an array with the results from the given function.
    pub fn from_fn<F: FnMut(D::Shape) -> T>(shape: D::Shape, f: F) -> Self {
        Self::from_fn_in(shape, f, Global)
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

macro_rules! impl_sub_grid {
    ($name:tt, $buffer:tt, $as_ptr:tt, $raw_mut:tt, {$($mut:tt)?}) => {
        impl<'a, T, L: Copy> $name<'a, T, L> {
            /// Creates an array view from a raw pointer and layout.
            /// # Safety
            /// The pointer must be non-null and a valid array view for the given layout.
            pub unsafe fn new_unchecked(ptr: *$raw_mut T, layout: L) -> Self {
                Self { buffer: $buffer::new_unchecked(ptr, layout) }
            }
        }

        impl<'a, T, D: Dim, F: Format> $name<'a, T, Layout<D, F>> {
            /// Converts the array view into a one-dimensional array view.
            /// # Panics
            /// Panics if the array layout is not uniformly strided.
            pub fn into_flattened(
                $($mut)? self
            ) -> $name<'a, T, Layout<Rank<1, D::Order>, F::Uniform>> {
                unsafe { $name::new_unchecked(self.$as_ptr(), self.layout().flatten()) }
            }

            /// Converts the array view into a reformatted array view.
            /// # Panics
            /// Panics if the array layout is not compatible with the new format.
            pub fn into_format<G: Format>($($mut)? self) -> $name<'a, T, Layout<D, G>> {
                unsafe { $name::new_unchecked(self.$as_ptr(), self.layout().reformat()) }
            }

            /// Converts the array view into a reshaped array view with similar layout.
            /// # Panics
            /// Panics if the array length is changed, or the memory layout is not compatible.
            pub fn into_shape<S: Shape>(
                $($mut)? self,
                shape: S
            ) -> $name<'a, T, ValidLayout<S::Dim<D::Order>, F>> {
                unsafe { $name::new_unchecked(self.$as_ptr(), self.layout().reshape(shape)) }
            }

            /// Divides an array view into two at an index along the outer dimension.
            /// # Panics
            /// Panics if the split point is larger than the number of elements in that dimension.
            pub fn into_split_at(
                self,
                mid: usize,
            ) -> ($name<'a, T, Layout<D, F>>, $name<'a, T, Layout<D, F>>) {
                assert!(D::RANK > 0, "invalid rank");

                self.into_split_dim_at(D::dim(D::RANK - 1), mid)
            }

            /// Divides an array view into two at an index along the specified dimension.
            /// # Panics
            /// Panics if the split point is larger than the number of elements in that dimension.
            pub fn into_split_axis_at<const DIM: usize>(
                self,
                mid: usize,
            ) -> (
                $name<'a, T, Layout<D, <Const<DIM> as Axis<D>>::Split<F>>>,
                $name<'a, T, Layout<D, <Const<DIM> as Axis<D>>::Split<F>>>
            )
            where
                Const<DIM>: Axis<D>
            {
                self.into_format().into_split_dim_at(DIM, mid)
            }

             /// Converts an array view into a new array view for the specified subarray.
            /// # Panics
            /// Panics if the subarray is out of bounds.
            pub fn into_view<I: ViewIndex<D, F>>(
                $($mut)? self,
                index: I
            ) -> $name<'a, T, ViewLayout<I::Params>> {
                let (offset, layout) = I::view_index(index, self.layout());
                let count = if layout.is_empty() { 0 } else { offset }; // Discard offset if empty.

                unsafe { $name::new_unchecked(self.$as_ptr().offset(count), layout) }
            }

            fn into_split_dim_at(
                $($mut)? self,
                dim: usize,
                mid: usize
            ) -> ($name<'a, T, Layout<D, F>>, $name<'a, T, Layout<D, F>>) {
                if mid > self.size(dim) {
                    panic_bounds_check(mid, self.size(dim));
                }

                let left_layout = self.layout().resize_dim(dim, mid);
                let right_layout = self.layout().resize_dim(dim, self.size(dim) - mid);

                // Calculate offset for the second view if non-empty.
                let count = if mid == self.size(dim) { 0 } else { self.stride(dim) * mid as isize };

                unsafe {
                    let left = $name::new_unchecked(self.$as_ptr(), left_layout);
                    let right = $name::new_unchecked(self.$as_ptr().offset(count), right_layout);

                    (left, right)
                }
            }
        }
    };
}

impl_sub_grid!(SubGrid, SubBuffer, as_ptr, const, {});
impl_sub_grid!(SubGridMut, SubBufferMut, as_mut_ptr, mut, {mut});

impl<T, D: Dim, A: Allocator> Borrow<DenseSpan<T, D>> for DenseGrid<T, D, A> {
    fn borrow(&self) -> &DenseSpan<T, D> {
        self.as_span()
    }
}

impl<T, D: Dim, A: Allocator> BorrowMut<DenseSpan<T, D>> for DenseGrid<T, D, A> {
    fn borrow_mut(&mut self) -> &mut DenseSpan<T, D> {
        self.as_mut_span()
    }
}

impl<B: Buffer + Clone> Clone for GridBase<B> {
    fn clone(&self) -> Self {
        Self { buffer: self.buffer.clone() }
    }

    fn clone_from(&mut self, src: &Self) {
        self.buffer.clone_from(&src.buffer);
    }
}

impl<B: Buffer> Debug for GridBase<B>
where
    SpanBase<B::Item, B::Layout>: Debug,
{
    fn fmt(&self, fmt: &mut Formatter<'_>) -> fmt::Result {
        self.as_span().fmt(fmt)
    }
}

impl<T, D: Dim> Default for DenseGrid<T, D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Buffer> Deref for GridBase<B> {
    type Target = SpanBase<B::Item, B::Layout>;

    fn deref(&self) -> &Self::Target {
        self.as_span()
    }
}

impl<B: BufferMut> DerefMut for GridBase<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_span()
    }
}

impl<'a, T: 'a + Copy, O: Order, A: 'a + Allocator> Extend<&'a T> for DenseGrid<T, Rank<1, O>, A> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        unsafe {
            self.buffer.vec_mut().extend(iter);
            self.buffer.set_layout(DenseLayout::new([self.buffer.vec().len()]));
        }
    }
}

impl<T, O: Order, A: Allocator> Extend<T> for DenseGrid<T, Rank<1, O>, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        unsafe {
            self.buffer.vec_mut().extend(iter);
            self.buffer.set_layout(DenseLayout::new([self.buffer.vec().len()]));
        }
    }
}

impl<T: Clone, O: Order> From<&[T]> for DenseGrid<T, Rank<1, O>> {
    fn from(slice: &[T]) -> Self {
        Self::from(slice.to_vec())
    }
}

impl<'a, T, O: Order> From<&'a [T]> for SubGrid<'a, T, DenseLayout<Rank<1, O>>> {
    fn from(slice: &'a [T]) -> Self {
        unsafe { SubGrid::new_unchecked(slice.as_ptr(), DenseLayout::new([slice.len()])) }
    }
}

impl<'a, T, O: Order> From<&'a mut [T]> for SubGridMut<'a, T, DenseLayout<Rank<1, O>>> {
    fn from(slice: &'a mut [T]) -> Self {
        unsafe { SubGridMut::new_unchecked(slice.as_mut_ptr(), DenseLayout::new([slice.len()])) }
    }
}

macro_rules! impl_from_array_ref {
    ($n:tt, ($($xyz:tt),+), ($($zyx:tt),+), $array:tt) => {
        impl<'a, T, O: Order, $(const $xyz: usize),+> From<&'a $array>
            for SubGrid<'a, T, DenseLayout<Rank<$n, O>>>
        {
            fn from(array: &'a $array) -> Self {
                let layout = DenseLayout::new(O::select([$($xyz),+], [$($zyx),+]));

                unsafe { Self::new_unchecked(array.as_ptr().cast(), layout) }
            }
        }

        impl<'a, T, O: Order, $(const $xyz: usize),+> From<&'a mut $array>
            for SubGridMut<'a, T, DenseLayout<Rank<$n, O>>>
        {
            fn from(array: &'a mut $array) -> Self {
                let layout = DenseLayout::new(O::select([$($xyz),+], [$($zyx),+]));

                unsafe { Self::new_unchecked(array.as_mut_ptr().cast(), layout) }
            }
        }
    };
}

impl_from_array_ref!(1, (X), (X), [T; X]);
impl_from_array_ref!(2, (X, Y), (Y, X), [[T; X]; Y]);
impl_from_array_ref!(3, (X, Y, Z), (Z, Y, X), [[[T; X]; Y]; Z]);
impl_from_array_ref!(4, (X, Y, Z, W), (W, Z, Y, X), [[[[T; X]; Y]; Z]; W]);
impl_from_array_ref!(5, (X, Y, Z, W, U), (U, W, Z, Y, X), [[[[[T; X]; Y]; Z]; W]; U]);
impl_from_array_ref!(6, (X, Y, Z, W, U, V), (V, U, W, Z, Y, X), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

macro_rules! impl_from_array {
    ($n:tt, ($($xyz:tt),+), ($($zyx:tt),+), $array:tt) => {
        #[allow(unused_parens)]
        impl<T, O: Order, $(const $xyz: usize),+> From<$array> for DenseGrid<T, Rank<$n, O>> {
            fn from(array: $array) -> Self {
                #[cfg(not(feature = "nightly"))]
                let mut vec = std::mem::ManuallyDrop::new(Vec::from(array));
                #[cfg(not(feature = "nightly"))]
                let (ptr, mut capacity) = (vec.as_mut_ptr(), vec.capacity());
                #[cfg(feature = "nightly")]
                let (ptr, _, mut capacity, alloc) = Vec::from(array).into_raw_parts_with_alloc();
                let layout = DenseLayout::new(O::select([$($xyz),+], [$($zyx),+]));

                unsafe {
                    capacity *= mem::size_of_val(&*ptr) / mem::size_of::<T>();

                    #[cfg(not(feature = "nightly"))]
                    let vec = Vec::from_raw_parts(ptr.cast(), layout.len(), capacity);
                    #[cfg(feature = "nightly")]
                    let vec = Vec::from_raw_parts_in(ptr.cast(), layout.len(), capacity, alloc);

                    Self::from_parts(vec, layout)
                }
            }
        }
    };
}

impl_from_array!(1, (X), (X), [T; X]);
impl_from_array!(2, (X, Y), (Y, X), [[T; X]; Y]);
impl_from_array!(3, (X, Y, Z), (Z, Y, X), [[[T; X]; Y]; Z]);
impl_from_array!(4, (X, Y, Z, W), (W, Z, Y, X), [[[[T; X]; Y]; Z]; W]);
impl_from_array!(5, (X, Y, Z, W, U), (U, W, Z, Y, X), [[[[[T; X]; Y]; Z]; W]; U]);
impl_from_array!(6, (X, Y, Z, W, U, V), (V, U, W, Z, Y, X), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

impl<T, O: Order, A: Allocator> From<vec_t!(T, A)> for DenseGrid<T, Rank<1, O>, A> {
    fn from(vec: vec_t!(T, A)) -> Self {
        let layout = DenseLayout::new([vec.len()]);

        unsafe { Self::from_parts(vec, layout) }
    }
}

impl<T, D: Dim, A: Allocator> From<DenseGrid<T, D, A>> for vec_t!(T, A) {
    fn from(grid: DenseGrid<T, D, A>) -> Self {
        grid.into_vec()
    }
}

impl<T, O: Order> FromIterator<T> for DenseGrid<T, Rank<1, O>> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Vec::from_iter(iter))
    }
}

impl<'a, B: Buffer> IntoIterator for &'a GridBase<B>
where
    &'a SpanBase<B::Item, B::Layout>: IntoIterator<Item = &'a B::Item>,
{
    type Item = &'a B::Item;
    type IntoIter = <&'a SpanBase<B::Item, B::Layout> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.as_span().into_iter()
    }
}

impl<'a, B: BufferMut> IntoIterator for &'a mut GridBase<B>
where
    &'a mut SpanBase<B::Item, B::Layout>: IntoIterator<Item = &'a mut B::Item>,
{
    type Item = &'a mut B::Item;
    type IntoIter = <&'a mut SpanBase<B::Item, B::Layout> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_span().into_iter()
    }
}

impl<T, D: Dim, A: Allocator> IntoIterator for DenseGrid<T, D, A> {
    type Item = T;
    type IntoIter = <vec_t!(T, A) as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

unsafe fn extend_from_span<T: Clone, F: Format, A: Allocator>(
    vec: &mut vec_t!(T, A),
    span: &SpanBase<T, Layout<impl Dim, F>>,
) {
    if F::IS_UNIFORM {
        for x in span.flatten().iter() {
            vec.as_mut_ptr().add(vec.len()).write(x.clone());
            vec.set_len(vec.len() + 1);
        }
    } else {
        for x in span.outer_iter() {
            #[cfg(not(feature = "nightly"))]
            extend_from_span::<_, _, A>(vec, &x);
            #[cfg(feature = "nightly")]
            extend_from_span(vec, &x);
        }
    }
}

unsafe fn from_fn<T, D: Dim, A: Allocator, I: Dim, F: FnMut(D::Shape) -> T>(
    vec: &mut vec_t!(T, A),
    shape: D::Shape,
    mut index: D::Shape,
    f: &mut F,
) {
    let dim = D::dim(I::RANK);

    for i in 0..shape[dim] {
        index[dim] = i;

        if I::RANK == 0 {
            vec.as_mut_ptr().add(vec.len()).write(f(index));
            vec.set_len(vec.len() + 1);
        } else {
            from_fn::<T, D, A, I::Lower, F>(vec, shape, index, f);
        }
    }
}

fn map<T: Default, F: Format>(
    span: &mut SpanBase<T, Layout<impl Dim, F>>,
    f: &mut impl FnMut(T) -> T,
) {
    if F::IS_UNIFORM {
        for x in span.flatten_mut().iter_mut() {
            *x = f(mem::take(x));
        }
    } else {
        for mut x in span.outer_iter_mut() {
            map(&mut x, f);
        }
    }
}
