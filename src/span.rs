#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::slice;

use crate::array::{GridArray, SpanArray, ViewArray, ViewArrayMut};
use crate::dim::{Const, Dim, Shape};
use crate::index::{Axis, DimIndex, Permutation, SpanIndex, ViewIndex};
use crate::iter::{AxisIter, AxisIterMut};
use crate::layout::{Dense, Layout, Uniform};
use crate::mapping::Mapping;
use crate::raw_span::RawSpan;

impl<T, D: Dim, L: Layout> SpanArray<T, D, L> {
    /// Returns a mutable pointer to the array buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if D::RANK > 0 {
            RawSpan::from_mut_span(self).as_mut_ptr()
        } else {
            self as *mut Self as *mut T
        }
    }

    /// Returns a raw pointer to the array buffer.
    pub fn as_ptr(&self) -> *const T {
        if D::RANK > 0 {
            RawSpan::from_span(self).as_ptr()
        } else {
            self as *const Self as *const T
        }
    }

    /// Returns an iterator that gives array views over the specified dimension.
    ///
    /// When iterating over the outer dimension, both the unit inner stride and the
    /// uniform stride properties are maintained, and the resulting array views have
    /// the same layout.
    ///
    /// When iterating over the inner dimension, the uniform stride property is
    /// maintained but not unit inner stride, and the resulting array views have
    /// flat or strided layout.
    ///
    /// When iterating over the middle dimensions, the unit inner stride propery is
    /// maintained but not uniform stride, and the resulting array views have general
    /// or strided layout.
    pub fn axis_iter<const DIM: usize>(
        &self,
    ) -> AxisIter<T, D::Lower, <Const<DIM> as Axis<D>>::Remove<L>>
    where
        Const<DIM>: Axis<D>,
    {
        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                Mapping::remove_dim(self.mapping(), DIM),
                self.size(DIM),
                self.stride(DIM),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the specified dimension.
    ///
    /// When iterating over the outer dimension, both the unit inner stride and the
    /// uniform stride properties are maintained, and the resulting array views have
    /// the same layout.
    ///
    /// When iterating over the inner dimension, the uniform stride property is
    /// maintained but not unit inner stride, and the resulting array views have
    /// flat or strided layout.
    ///
    /// When iterating over the middle dimensions, the unit inner stride propery is
    /// maintained but not uniform stride, and the resulting array views have general
    /// or strided layout.
    pub fn axis_iter_mut<const DIM: usize>(
        &mut self,
    ) -> AxisIterMut<T, D::Lower, <Const<DIM> as Axis<D>>::Remove<L>>
    where
        Const<DIM>: Axis<D>,
    {
        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                Mapping::remove_dim(self.mapping(), DIM),
                self.size(DIM),
                self.stride(DIM),
            )
        }
    }

    /// Clones an array span into the array span.
    ///
    /// # Panics
    ///
    /// Panics if the two spans have different shapes.
    pub fn clone_from_span(&mut self, src: &SpanArray<T, D, impl Layout>)
    where
        T: Clone,
    {
        clone_from_span(self, src);
    }

    /// Returns `true` if the array span contains an element with the given value.
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        contains(self, x)
    }

    /// Fills the array span with elements by cloning `value`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        fill_with(self, &mut || value.clone());
    }

    /// Fills the array span with elements returned by calling a closure repeatedly.
    pub fn fill_with(&mut self, mut f: impl FnMut() -> T) {
        fill_with(self, &mut f);
    }

    /// Returns a one-dimensional array view of the array span.
    ///
    /// # Panics
    ///
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten(&self) -> ViewArray<T, Const<1>, L::Uniform> {
        self.to_view().into_flattened()
    }

    /// Returns a mutable one-dimensional array view over the array span.
    ///
    /// # Panics
    ///
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten_mut(&mut self) -> ViewArrayMut<T, Const<1>, L::Uniform> {
        self.to_view_mut().into_flattened()
    }

    /// Returns a reference to an element or a subslice, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// The index must be within bounds of the array span.
    pub unsafe fn get_unchecked<I: SpanIndex<T, D, L>>(&self, index: I) -> &I::Output {
        index.get_unchecked(self)
    }

    /// Returns a mutable reference to an element or a subslice, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// The index must be within bounds of the array span.
    pub unsafe fn get_unchecked_mut<I: SpanIndex<T, D, L>>(&mut self, index: I) -> &mut I::Output {
        index.get_unchecked_mut(self)
    }

    /// Returns an iterator that gives array views over the inner dimension.
    ///
    /// Iterating over the inner dimension maintains the uniform stride property but not
    /// unit inner stride, so that the resulting array views have flat or strided layout.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn inner_iter(
        &self,
    ) -> AxisIter<T, D::Lower, <D::Lower as Dim>::Layout<L::NonUnitStrided>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                Mapping::remove_dim(self.mapping(), 0),
                self.size(0),
                self.stride(0),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the inner dimension.
    ///
    /// Iterating over the inner dimension maintains the uniform stride property but not
    /// unit inner stride, so that the resulting array views have flat or strided layout.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn inner_iter_mut(
        &mut self,
    ) -> AxisIterMut<T, D::Lower, <D::Lower as Dim>::Layout<L::NonUnitStrided>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                Mapping::remove_dim(self.mapping(), 0),
                self.size(0),
                self.stride(0),
            )
        }
    }

    /// Returns `true` if the array strides are consistent with contiguous memory layout.
    pub fn is_contiguous(&self) -> bool {
        self.mapping().is_contiguous()
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.mapping().is_empty()
    }

    /// Returns `true` if the array strides are consistent with uniformly strided memory layout.
    pub fn is_uniformly_strided(&self) -> bool {
        self.mapping().is_uniformly_strided()
    }

    /// Returns an iterator over the array span, which must support linear indexing.
    pub fn iter(&self) -> L::Iter<'_, T>
    where
        L: Uniform,
    {
        L::Mapping::iter(self)
    }

    /// Returns a mutable iterator over the array span, which must support linear indexing.
    pub fn iter_mut(&mut self) -> L::IterMut<'_, T>
    where
        L: Uniform,
    {
        L::Mapping::iter_mut(self)
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.mapping().len()
    }

    /// Returns the array layout mapping.
    pub fn mapping(&self) -> L::Mapping<D> {
        if D::RANK > 0 {
            RawSpan::from_span(self).mapping()
        } else {
            L::Mapping::default()
        }
    }

    /// Returns an iterator that gives array views over the outer dimension.
    ///
    /// Iterating over the outer dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same layout.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn outer_iter(&self) -> AxisIter<T, D::Lower, <D::Lower as Dim>::Layout<L>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIter::new_unchecked(
                self.as_ptr(),
                Mapping::remove_dim(self.mapping(), D::RANK - 1),
                self.size(D::RANK - 1),
                self.stride(D::RANK - 1),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the outer dimension.
    ///
    /// Iterating over the outer dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same layout.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn outer_iter_mut(&mut self) -> AxisIterMut<T, D::Lower, <D::Lower as Dim>::Layout<L>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisIterMut::new_unchecked(
                self.as_mut_ptr(),
                Mapping::remove_dim(self.mapping(), D::RANK - 1),
                self.size(D::RANK - 1),
                self.stride(D::RANK - 1),
            )
        }
    }

    /// Returns the array rank, i.e. the number of dimensions.
    pub fn rank(&self) -> usize {
        D::RANK
    }

    /// Returns a remapped array view of the array span.
    ///
    /// # Panics
    ///
    /// Panics if the memory layout is not compatible with the new array layout.
    pub fn remap<M: Layout>(&self) -> ViewArray<T, D, M> {
        self.to_view().into_mapping()
    }

    /// Returns a mutable remapped array view of the array span.
    ///
    /// # Panics
    ///
    /// Panics if the memory layout is not compatible with the new array layout.
    pub fn remap_mut<M: Layout>(&mut self) -> ViewArrayMut<T, D, M> {
        self.to_view_mut().into_mapping()
    }

    /// Returns a reshaped array view of the array span, with similar layout.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed, or the memory layout is not compatible.
    pub fn reshape<S: Shape>(&self, shape: S) -> ViewArray<T, S::Dim, <S::Dim as Dim>::Layout<L>> {
        self.to_view().into_shape(shape)
    }

    /// Returns a mutable reshaped array view of the array span, with similar layout.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed, or the memory layout is not compatible.
    pub fn reshape_mut<S: Shape>(
        &mut self,
        shape: S,
    ) -> ViewArrayMut<T, S::Dim, <S::Dim as Dim>::Layout<L>> {
        self.to_view_mut().into_shape(shape)
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> D::Shape {
        self.mapping().shape()
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn size(&self, dim: usize) -> usize {
        self.mapping().size(dim)
    }

    /// Divides an array span into two at an index along the outer dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_at(&self, mid: usize) -> (ViewArray<T, D, L>, ViewArray<T, D, L>) {
        self.to_view().into_split_at(mid)
    }

    /// Divides a mutable array span into two at an index along the outer dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_at_mut(&mut self, mid: usize) -> (ViewArrayMut<T, D, L>, ViewArrayMut<T, D, L>) {
        self.to_view_mut().into_split_at(mid)
    }

    /// Divides an array span into two at an index along the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_axis_at<const DIM: usize>(
        &self,
        mid: usize,
    ) -> (
        ViewArray<T, D, <Const<DIM> as Axis<D>>::Split<L>>,
        ViewArray<T, D, <Const<DIM> as Axis<D>>::Split<L>>,
    )
    where
        Const<DIM>: Axis<D>,
    {
        self.to_view().into_split_axis_at(mid)
    }

    /// Divides a mutable array span into two at an index along the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_axis_at_mut<const DIM: usize>(
        &mut self,
        mid: usize,
    ) -> (
        ViewArrayMut<T, D, <Const<DIM> as Axis<D>>::Split<L>>,
        ViewArrayMut<T, D, <Const<DIM> as Axis<D>>::Split<L>>,
    )
    where
        Const<DIM>: Axis<D>,
    {
        self.to_view_mut().into_split_axis_at(mid)
    }

    /// Returns the distance between elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn stride(&self, dim: usize) -> isize {
        self.mapping().stride(dim)
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> D::Strides {
        self.mapping().strides()
    }

    /// Copies the array span into a new array.
    #[cfg(not(feature = "nightly"))]
    pub fn to_grid(&self) -> GridArray<T, D>
    where
        T: Clone,
    {
        let mut grid = GridArray::new();

        grid.extend_from_span(self);
        grid
    }

    /// Copies the array span into a new array.
    #[cfg(feature = "nightly")]
    pub fn to_grid(&self) -> GridArray<T, D>
    where
        T: Clone,
    {
        self.to_grid_in(Global)
    }

    /// Copies the array span into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> GridArray<T, D, A>
    where
        T: Clone,
    {
        let mut grid = GridArray::new_in(alloc);

        grid.extend_from_span(self);
        grid
    }

    /// Copies the array span into a new vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_grid().into_vec()
    }

    /// Copies the array span into a new vector with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_vec_in<A: Allocator>(&self, alloc: A) -> Vec<T, A>
    where
        T: Clone,
    {
        self.to_grid_in(alloc).into_vec()
    }

    /// Returns an array view of the entire array span.
    pub fn to_view(&self) -> ViewArray<T, D, L> {
        unsafe { ViewArray::new_unchecked(self.as_ptr(), self.mapping()) }
    }

    /// Returns a mutable array view of the entire array span.
    pub fn to_view_mut(&mut self) -> ViewArrayMut<T, D, L> {
        unsafe { ViewArrayMut::new_unchecked(self.as_mut_ptr(), self.mapping()) }
    }
}

impl<T, D: Dim> SpanArray<T, D, Dense> {
    /// Returns a mutable slice of all elements in the array.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Returns a slice of all elements in the array.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

macro_rules! impl_permute {
    ($n:tt, ($($xyz:tt),+)) => {
        impl<T, L: Layout> SpanArray<T, Const<$n>, L> {
            /// Returns an array view with the dimensions permuted.
            pub fn permute<$(const $xyz: usize),+>(
                &self
            ) -> ViewArray<T, Const<$n>, <($(Const<$xyz>,)+) as Permutation>::Layout<L>>
            where
                ($(Const<$xyz>,)+): Permutation
            {
                self.to_view().into_permuted()
            }

            /// Returns a mutable array view with the dimensions permuted.
            pub fn permute_mut<$(const $xyz: usize),+>(
                &mut self
            ) -> ViewArrayMut<T, Const<$n>, <($(Const<$xyz>,)+) as Permutation>::Layout<L>>
            where
                ($(Const<$xyz>,)+): Permutation
            {
                self.to_view_mut().into_permuted()
            }
        }
    };
}

impl_permute!(1, (X));
impl_permute!(2, (X, Y));
impl_permute!(3, (X, Y, Z));
impl_permute!(4, (X, Y, Z, W));
impl_permute!(5, (X, Y, Z, W, U));
impl_permute!(6, (X, Y, Z, W, U, V));

macro_rules! impl_view {
    ($n:tt, ($($xyz:tt),+), ($($idx:tt),+)) => {
        #[allow(unused_parens)]
        impl<T, L: Layout> SpanArray<T, Const<$n>, L> {
            /// Copies the specified subarray into a new array.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn grid<$($xyz: DimIndex),+>(
                &self,
                $($idx: $xyz),+
            ) -> GridArray<T, <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim>
            where
                T: Clone,
            {
                self.view($($idx),+).to_grid()
            }

            /// Returns an array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn view<$($xyz: DimIndex),+>(
                &self,
                $($idx: $xyz),+
            ) -> ViewArray<
                T,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Layout,
            > {
                self.to_view().into_view($($idx),+)
            }

            /// Returns a mutable array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn view_mut<$($xyz: DimIndex),+>(
                &mut self,
                $($idx: $xyz),+,
            ) -> ViewArrayMut<
                T,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Layout,
            > {
                self.to_view_mut().into_view($($idx),+)
            }
        }
    };
}

impl_view!(1, (X), (x));
impl_view!(2, (X, Y), (x, y));
impl_view!(3, (X, Y, Z), (x, y, z));
impl_view!(4, (X, Y, Z, W), (x, y, z, w));
impl_view!(5, (X, Y, Z, W, U), (x, y, z, w, u));
impl_view!(6, (X, Y, Z, W, U, V), (x, y, z, w, u, v));

impl<T: Clone, D: Dim> ToOwned for SpanArray<T, D, Dense> {
    type Owned = GridArray<T, D>;

    fn to_owned(&self) -> Self::Owned {
        self.to_grid()
    }

    fn clone_into(&self, target: &mut Self::Owned) {
        unsafe {
            target.buffer.with_mut_vec(|vec| {
                self.as_slice().clone_into(vec);
            });

            target.buffer.set_mapping(self.mapping());
        }
    }
}

fn clone_from_span<T: Clone, D: Dim, L: Layout, M: Layout>(
    this: &mut SpanArray<T, D, L>,
    src: &SpanArray<T, D, M>,
) {
    if L::IS_UNIFORM && M::IS_UNIFORM {
        assert!(src.shape()[..] == this.shape()[..], "shape mismatch");

        if L::IS_UNIT_STRIDED && M::IS_UNIT_STRIDED {
            this.remap_mut().as_mut_slice().clone_from_slice(src.remap().as_slice());
        } else {
            for (x, y) in this.flatten_mut().iter_mut().zip(src.flatten().iter()) {
                x.clone_from(y);
            }
        }
    } else {
        assert!(src.size(D::RANK - 1) == this.size(D::RANK - 1), "shape mismatch");

        for (mut x, y) in this.outer_iter_mut().zip(src.outer_iter()) {
            clone_from_span(&mut x, &y);
        }
    }
}

fn contains<T: PartialEq, D: Dim, L: Layout>(this: &SpanArray<T, D, L>, value: &T) -> bool {
    if L::IS_UNIFORM {
        if L::IS_UNIT_STRIDED {
            this.remap().as_slice().contains(value)
        } else {
            this.as_span().flatten().iter().any(|x| x == value)
        }
    } else {
        this.outer_iter().any(|x| x.contains(value))
    }
}

fn fill_with<T, L: Layout>(this: &mut SpanArray<T, impl Dim, L>, f: &mut impl FnMut() -> T) {
    if L::IS_UNIFORM {
        for x in this.flatten_mut().iter_mut() {
            *x = f();
        }
    } else {
        for mut x in this.outer_iter_mut() {
            fill_with(&mut x, f);
        }
    }
}
