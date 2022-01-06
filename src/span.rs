use std::alloc::{Allocator, Global};
use std::fmt::{Debug, Formatter, Result};
use std::marker::PhantomData;
use std::{mem, ptr, slice};

use crate::dimension::{Const, Dim};
use crate::format::UnitStrided;
use crate::grid::{DenseGrid, SubGrid, SubGridMut};
use crate::index::ViewIndex;
use crate::iterator::{AxisIter, AxisIterMut};
use crate::layout::{DenseLayout, Layout, StaticLayout, StridedLayout};
use crate::mapping::Mapping;
use crate::order::Order;

/// Multidimensional array span with static rank and element order.
#[repr(transparent)]
pub struct SpanBase<T, L: Layout> {
    _marker: PhantomData<(T, L)>,
    _slice: [()],
}

impl<T, L: Layout> SpanBase<T, L> {
    /// Returns a mutable pointer to the array buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        (self as *mut Self).cast()
    }

    /// Returns a raw pointer to the array buffer.
    pub fn as_ptr(&self) -> *const T {
        (self as *const Self).cast()
    }

    /// Returns a mutable slice of all elements in the array.
    /// # Panics
    /// Panics if the array layout is not contiguous.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert!(self.is_contiguous(), "array layout not contiguous");

        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Returns a slice of all elements in the array.
    /// # Panics
    /// Panics if the array layout is not contiguous.
    pub fn as_slice(&self) -> &[T] {
        assert!(self.is_contiguous(), "array layout not contiguous");

        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Returns an iterator that gives non-dense array views over the specified dimension.
    /// # Panics
    /// Panics if the inner or outer dimension is specified, as that would affect the return type.
    pub fn axis_iter(
        &self,
        dim: usize,
    ) -> AxisIter<T, <L::NonDense as Mapping<L::NonDense>>::Smaller> {
        assert!(dim > 0 && dim + 1 < self.rank(), "inner or outer dimension not allowed");

        unsafe {
            AxisIter::new(
                self.as_ptr(),
                self.layout().to_non_dense().remove_dim(dim),
                self.size(self.dim(dim)),
                self.stride(self.dim(dim)),
            )
        }
    }

    /// Returns a mutable iterator that gives non-dense array views over the specified dimension.
    /// # Panics
    /// Panics if the inner or outer dimension is specified, as that would affect the return type.
    pub fn axis_iter_mut(
        &mut self,
        dim: usize,
    ) -> AxisIterMut<T, <L::NonDense as Mapping<L::NonDense>>::Smaller> {
        assert!(dim > 0 && dim + 1 < self.rank(), "inner or outer dimension not allowed");

        unsafe {
            AxisIterMut::new(
                self.as_mut_ptr(),
                self.layout().to_non_dense().remove_dim(dim),
                self.size(self.dim(dim)),
                self.stride(self.dim(dim)),
            )
        }
    }

    /// Returns the dimension with the specified index, counted from the innermost dimension.
    pub fn dim(&self, index: usize) -> usize {
        self.layout().dim(index)
    }

    /// Fills the array span with elements by cloning `value`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        fill(self, &value);
    }

    /// Fills the array span with elements returned by calling a closure repeatedly.
    pub fn fill_with<F: FnMut() -> T>(&mut self, mut f: F) {
        fill_with(self, &mut f);
    }

    /// Returns an iterator over the flattened array span.
    /// # Panics
    /// Panics if the array layout is not compatible with linear indexing and fixed stride.
    pub fn flat_iter(&self) -> L::Iter<'_, T> {
        L::iter(self)
    }

    /// Returns a mutable iterator over the flattened array span.
    /// # Panics
    /// Panics if the array layout is not compatible with linear indexing and fixed stride.
    pub fn flat_iter_mut(&mut self) -> L::IterMut<'_, T> {
        L::iter_mut(self)
    }

    /// Creates an array span from a raw pointer and an array layout.
    /// # Safety
    /// The pointer must be a valid array span for the given layout.
    pub unsafe fn from_raw_parts(ptr: *const T, layout: &L) -> *const Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        if mem::size_of::<L>() == 0 {
            ptr::from_raw_parts(ptr.cast(), 0usize)
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            ptr::from_raw_parts(ptr.cast(), mem::transmute_copy(layout))
        } else {
            ptr::from_raw_parts(ptr.cast(), layout as *const L as usize)
        }
    }

    /// Creates a mutable array span from a raw pointer and an array layout.
    /// # Safety
    /// The pointer must be a valid array span for the given layout.
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, layout: &L) -> *mut Self {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        if mem::size_of::<L>() == 0 {
            ptr::from_raw_parts_mut(ptr.cast(), 0usize)
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            ptr::from_raw_parts_mut(ptr.cast(), mem::transmute_copy(layout))
        } else {
            ptr::from_raw_parts_mut(ptr.cast(), layout as *const L as usize)
        }
    }

    /// Returns an iterator that gives strided array views over the inner dimension.
    /// # Panics
    /// Panics if the rank is not 2 or higher.
    pub fn inner_iter(&self) -> AxisIter<T, StridedLayout<<L::Dim as Dim>::Smaller, L::Order>> {
        assert!(self.rank() > 1, "rank must be 2 or higher");

        unsafe {
            AxisIter::new(
                self.as_ptr(),
                self.layout().to_strided().remove_dim(0),
                self.size(self.dim(0)),
                self.stride(self.dim(0)),
            )
        }
    }

    /// Returns a mutable iterator that gives strided array views over the inner dimension.
    /// # Panics
    /// Panics if the rank is not 2 or higher.
    pub fn inner_iter_mut(
        &mut self,
    ) -> AxisIterMut<T, StridedLayout<<L::Dim as Dim>::Smaller, L::Order>> {
        assert!(self.rank() > 1, "rank must be 2 or higher");

        unsafe {
            AxisIterMut::new(
                self.as_mut_ptr(),
                self.layout().to_strided().remove_dim(0),
                self.size(self.dim(0)),
                self.stride(self.dim(0)),
            )
        }
    }

    /// Returns true if the array has column-major element order.
    pub fn is_column_major(&self) -> bool {
        L::Order::select(true, false)
    }

    /// Returns true if the array elements are stored contiguously in memory.
    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Returns true if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.layout().is_empty()
    }

    /// Returns true if the array has row-major element order.
    pub fn is_row_major(&self) -> bool {
        L::Order::select(false, true)
    }

    /// Returns the array layout.
    pub fn layout(&self) -> L {
        let layout = ptr::metadata(self);

        if mem::size_of::<L>() == 0 {
            unsafe { mem::transmute_copy(&()) }
        } else if mem::size_of::<L>() == mem::size_of::<usize>() {
            unsafe { mem::transmute_copy(&layout) }
        } else {
            unsafe { *(layout as *const L) }
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.layout().len()
    }

    /// Returns an iterator that gives array views over the outer dimension.
    /// # Panics
    /// Panics if the rank is not 2 or higher.
    pub fn outer_iter(&self) -> AxisIter<T, L::Smaller> {
        assert!(self.rank() > 1, "rank must be 2 or higher");

        unsafe {
            AxisIter::new(
                self.as_ptr(),
                self.layout().remove_dim(self.dim(self.rank() - 1)),
                self.size(self.dim(self.rank() - 1)),
                self.stride(self.dim(self.rank() - 1)),
            )
        }
    }

    /// Returns a mutable iterator that gives array views over the outer dimension.
    /// # Panics
    /// Panics if the rank is not 2 or higher.
    pub fn outer_iter_mut(&mut self) -> AxisIterMut<T, L::Smaller> {
        assert!(self.rank() > 1, "rank must be 2 or higher");

        unsafe {
            AxisIterMut::new(
                self.as_mut_ptr(),
                self.layout().remove_dim(self.dim(self.rank() - 1)),
                self.size(self.dim(self.rank() - 1)),
                self.stride(self.dim(self.rank() - 1)),
            )
        }
    }

    /// Returns the rank of the array.
    pub fn rank(&self) -> usize {
        self.layout().rank()
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> <L::Dim as Dim>::Shape {
        self.layout().shape()
    }

    /// Returns the number of elements in the specified dimension.
    pub fn size(&self, dim: usize) -> usize {
        self.layout().size(dim)
    }

    /// Returns the distance between elements in the specified dimension.
    pub fn stride(&self, dim: usize) -> isize {
        self.layout().stride(dim)
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> <L::Dim as Dim>::Strides {
        self.layout().strides()
    }

    /// Copies the array span into a new array.
    pub fn to_grid(&self) -> DenseGrid<T, L::Dim, L::Order>
    where
        T: Clone,
    {
        self.to_grid_in(Global)
    }

    /// Copies the array span into a new array with the specified allocator.
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> DenseGrid<T, L::Dim, L::Order, A>
    where
        T: Clone,
    {
        DenseGrid::from(self.to_vec_in(alloc)).reshape(self.shape())
    }

    /// Copies the array span into a new vector with the specified allocator.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_vec_in(Global)
    }

    /// Copies the array span into a new vector with the specified allocator.
    pub fn to_vec_in<A: Allocator>(&self, alloc: A) -> Vec<T, A>
    where
        T: Clone,
    {
        let mut vec = Vec::with_capacity_in(self.len(), alloc);

        extend(self, &mut vec);

        vec
    }

    /// Returns an array view for the specified subarray.
    pub fn view<I>(&self, index: I) -> SubGrid<T, I::Layout>
    where
        I: ViewIndex<L::Dim, L::Order, L>,
    {
        let (offset, layout, _) = I::view_info(index, self.layout());

        unsafe { SubGrid::new(self.as_ptr().offset(offset), layout) }
    }

    /// Returns a mutable array view for the specified subarray.
    pub fn view_mut<I>(&mut self, index: I) -> SubGridMut<T, I::Layout>
    where
        I: ViewIndex<L::Dim, L::Order, L>,
    {
        let (offset, layout, _) = I::view_info(index, self.layout());

        unsafe { SubGridMut::new(self.as_mut_ptr().offset(offset), layout) }
    }

    /// Returns an array view for the entire array span.
    pub fn to_view(&self) -> SubGrid<T, L> {
        unsafe { SubGrid::new(self.as_ptr(), self.layout()) }
    }

    /// Returns a mutable array view for the entire array span.
    pub fn to_view_mut(&mut self) -> SubGridMut<T, L> {
        unsafe { SubGridMut::new(self.as_mut_ptr(), self.layout()) }
    }
}

impl<T, L: Layout<Dim = Const<1>>> SpanBase<T, L> {
    /// Returns an iterator over the one-dimensional array span.
    pub fn iter(&self) -> L::Iter<'_, T> {
        L::iter(self)
    }

    /// Returns a mutable iterator over the one-dimensional array span.
    pub fn iter_mut(&mut self) -> L::IterMut<'_, T> {
        L::iter_mut(self)
    }
}

impl<T, F: UnitStrided, L: Layout<Dim = Const<1>, Format = F>> AsMut<[T]> for SpanBase<T, L> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, F: UnitStrided, L: Layout<Dim = Const<1>, Format = F>> AsRef<[T]> for SpanBase<T, L> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, O: Order> AsMut<SpanBase<T, DenseLayout<Const<1>, O>>> for [T] {
    fn as_mut(&mut self) -> &mut SpanBase<T, DenseLayout<Const<1>, O>> {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        unsafe { &mut *ptr::from_raw_parts_mut(self.as_mut_ptr().cast(), self.len()) }
    }
}

impl<T, O: Order> AsRef<SpanBase<T, DenseLayout<Const<1>, O>>> for [T] {
    fn as_ref(&self) -> &SpanBase<T, DenseLayout<Const<1>, O>> {
        assert!(mem::size_of::<T>() != 0, "ZST not allowed");

        unsafe { &*ptr::from_raw_parts(self.as_ptr().cast(), self.len()) }
    }
}

macro_rules! impl_as_mut_ref_array {
    ($n:tt, ($($xyz:tt),+), ($($zyx:tt),+), $array:tt) => {
        #[allow(unused_parens)]
        impl<T, O: Order, $(const $xyz: usize),+> AsMut<SpanBase<T, DenseLayout<Const<$n>, O>>>
            for $array
        {
            fn as_mut(&mut self) -> &mut SpanBase<T, DenseLayout<Const<$n>, O>> {
                let layout = O::select(
                    &<($(Const<$xyz>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                    &<($(Const<$zyx>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                );

                unsafe { &mut *SpanBase::from_raw_parts_mut(self.as_mut_ptr().cast(), layout) }
            }
        }

        #[allow(unused_parens)]
        impl<T, O: Order, $(const $xyz: usize),+> AsRef<SpanBase<T, DenseLayout<Const<$n>, O>>>
            for $array
        {
            fn as_ref(&self) -> &SpanBase<T, DenseLayout<Const<$n>, O>> {
                let layout = O::select(
                    &<($(Const<$xyz>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                    &<($(Const<$zyx>),+) as StaticLayout<Const<$n>, O>>::LAYOUT,
                );

                unsafe { &*SpanBase::from_raw_parts(self.as_ptr().cast(), layout) }
            }
        }
    };
}

impl_as_mut_ref_array!(1, (X), (X), [T; X]);
impl_as_mut_ref_array!(2, (X, Y), (Y, X), [[T; X]; Y]);
impl_as_mut_ref_array!(3, (X, Y, Z), (Z, Y, X), [[[T; X]; Y]; Z]);
impl_as_mut_ref_array!(4, (X, Y, Z, W), (W, Z, Y, X), [[[[T; X]; Y]; Z]; W]);
impl_as_mut_ref_array!(5, (X, Y, Z, W, U), (U, W, Z, Y, X), [[[[[T; X]; Y]; Z]; W]; U]);
impl_as_mut_ref_array!(6, (X, Y, Z, W, U, V), (V, U, W, Z, Y, X), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

impl<T: Debug, L: Layout> Debug for SpanBase<T, L> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        if L::Dim::RANK == 0 {
            self[<L::Dim as Dim>::Shape::default()].fmt(fmt)
        } else if L::Dim::RANK == 1 {
            fmt.debug_list().entries(self.flat_iter()).finish()
        } else {
            fmt.debug_list().entries(self.outer_iter()).finish()
        }
    }
}

impl<'a, T, L: Layout<Dim = Const<1>>> IntoIterator for &'a SpanBase<T, L> {
    type Item = &'a T;
    type IntoIter = L::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, L: Layout<Dim = Const<1>>> IntoIterator for &'a mut SpanBase<T, L> {
    type Item = &'a mut T;
    type IntoIter = L::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Clone, D: Dim, O: Order> ToOwned for SpanBase<T, DenseLayout<D, O>> {
    type Owned = DenseGrid<T, D, O>;

    fn to_owned(&self) -> Self::Owned {
        self.to_grid()
    }
}

fn extend<T: Clone, L: Layout, A: Allocator>(span: &SpanBase<T, L>, vec: &mut Vec<T, A>) {
    if L::IS_DENSE || (L::IS_UNIT_STRIDED && L::Dim::RANK < 2) || L::Dim::RANK == 0 {
        vec.extend_from_slice(span.as_slice());
    } else if L::Dim::RANK == 1 {
        vec.extend(span.flat_iter().cloned());
    } else {
        for x in span.outer_iter() {
            extend(&x, vec);
        }
    }
}

fn fill<T: Clone, L: Layout>(span: &mut SpanBase<T, L>, value: &T) {
    if L::IS_DENSE || (L::IS_UNIT_STRIDED && L::Dim::RANK < 2) || L::Dim::RANK == 0 {
        span.as_mut_slice().fill(value.clone());
    } else if L::Dim::RANK == 1 {
        for x in span.flat_iter_mut() {
            x.clone_from(value);
        }
    } else {
        for mut x in span.outer_iter_mut() {
            fill(&mut x, value);
        }
    }
}

fn fill_with<T, L: Layout, F: FnMut() -> T>(span: &mut SpanBase<T, L>, f: &mut F) {
    if L::IS_DENSE || L::Dim::RANK < 2 {
        for x in span.flat_iter_mut() {
            *x = f();
        }
    } else {
        for mut x in span.outer_iter_mut() {
            fill_with(&mut x, f);
        }
    }
}
