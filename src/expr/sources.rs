#[cfg(feature = "nightly")]
use std::alloc::Allocator;
use std::fmt::{Debug, Formatter, Result};
use std::marker::PhantomData;
use std::ptr::{self, NonNull};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::{GridArray, ViewArray, ViewArrayMut};
use crate::dim::{Const, Dim, Shape};
use crate::expr::Producer;
use crate::expression::Expression;
use crate::layout::{Dense, Layout, Strided};
use crate::mapping::{FlatMapping, Mapping, StridedMapping};

/// Array axis expression.
pub struct AxisExpr<'a, T, D: Dim, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<D::Lower>,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a T>,
}

/// Mutable array axis expression.
pub struct AxisExprMut<'a, T, D: Dim, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<D::Lower>,
    size: usize,
    stride: isize,
    phantom: PhantomData<&'a mut T>,
}

/// Expression that moves out of an array range.
pub struct Drain<'a, T, D: Dim, A: Allocator> {
    grid: &'a mut GridArray<T, D, A>,
    start: usize,
    end: usize,
    tail: usize,
    inner_len: usize,
    index: usize,
}

/// Expression that repeats an element by cloning.
#[derive(Clone)]
pub struct Fill<T> {
    value: T,
}

/// Expression that gives elements by calling a closure repeatedly.
#[derive(Clone)]
pub struct FillWith<F> {
    f: F,
}

/// Expression with a defined shape that repeats an element by cloning.
#[derive(Clone)]
pub struct FromElem<S, T> {
    shape: S,
    elem: T,
}

/// Expression with a defined shape and elements from the given function.
#[derive(Clone)]
pub struct FromFn<S, F> {
    shape: S,
    f: F,
    index: S,
}

/// Expression that moves out of an array.
pub struct IntoExpr<T, D: Dim, A: Allocator> {
    grid: GridArray<T, D, A>,
    index: usize,
}

/// Expression that gives references to array elements.
pub struct Expr<'a, T, D: Dim, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<D>,
    phantom: PhantomData<&'a T>,
}

/// Expression that gives mutable references to array elements.
pub struct ExprMut<'a, T, D: Dim, L: Layout> {
    ptr: NonNull<T>,
    mapping: L::Mapping<D>,
    phantom: PhantomData<&'a mut T>,
}

/// Creates an expression with elements by cloning `value`.
///
/// # Examples
///
/// ```
/// use mdarray::{grid, view, Grid, View,expr};
///
/// let mut g = grid![0; 3];
///
/// g.assign(expr::fill(1));
///
/// assert_eq!(g, view![1; 3]);
/// ```
pub fn fill<T: Clone>(value: T) -> Expression<Fill<T>> {
    Expression::new(Fill::new(value))
}

/// Creates an expression with elements returned by calling a closure repeatedly.
///
/// # Examples
///
/// ```
/// use mdarray::{grid, view, Grid, View,expr};
///
/// let mut g = grid![0; 3];
///
/// g.assign(expr::fill_with(|| 1));
///
/// assert_eq!(g, view![1; 3]);
/// ```
pub fn fill_with<T, F: FnMut() -> T>(f: F) -> Expression<FillWith<F>> {
    Expression::new(FillWith::new(f))
}

/// Creates an expression with the given shape and elements by cloning `value`.
///
/// # Examples
///
/// ```
/// use mdarray::{view, View,expr};
///
/// assert_eq!(expr::from_elem([2, 3], 1).eval(), view![[1; 2]; 3]);
/// ```
pub fn from_elem<S: Shape, T: Clone>(shape: S, elem: T) -> Expression<FromElem<S, T>> {
    Expression::new(FromElem::new(shape, elem))
}

/// Creates an expression with the given shape and elements from the given function.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, view, View};
///
/// assert_eq!(expr::from_fn([2, 3], |[i, j]| 2 * j + i).eval(), view![[0, 1], [2, 3], [4, 5]]);
/// ```
pub fn from_fn<T, S: Shape, F: FnMut(S) -> T>(shape: S, f: F) -> Expression<FromFn<S, F>> {
    Expression::new(FromFn::new(shape, f))
}

macro_rules! impl_axis_expr {
    ($name:tt, $view:tt, $raw_mut:tt, $repeatable:tt) => {
        impl<'a, T, D: Dim, L: Layout> $name<'a, T, D, L> {
            pub(crate) unsafe fn new_unchecked(
                ptr: *$raw_mut T,
                mapping: L::Mapping<D::Lower>,
                size: usize,
                stride: isize,
            ) -> Self {
                Self {
                    ptr: NonNull::new_unchecked(ptr as *mut T),
                    mapping,
                    size,
                    stride: if mapping.is_empty() { 0 } else { stride },
                    phantom: PhantomData
                }
            }
        }

        impl<'a, T: Debug, D: Dim, L: Layout> Debug for $name<'a, T, D, L> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let mut shape = D::Shape::default();
                let mut strides = D::Strides::default();

                shape[..D::RANK - 1].copy_from_slice(&self.mapping.shape()[..]);
                shape[D::RANK - 1] = self.size;

                strides[..D::RANK - 1].copy_from_slice(&self.mapping.strides()[..]);
                strides[D::RANK - 1] = self.stride;

                let mapping = if D::RANK > 1 {
                    Mapping::remap(StridedMapping::new(shape, strides))
                } else {
                    Mapping::remap(FlatMapping::new(shape, strides[0]))
                };

                let view = unsafe {
                    ViewArray::<T, D, D::Layout<Strided>>::new_unchecked(self.ptr.as_ptr(), mapping)
                };

                f.debug_tuple(stringify!($name)).field(&view).finish()
            }
        }

         impl<'a, T, D: Dim, L: Layout> Producer for $name<'a, T, D, L> {
            type Item = $view<'a, T, D::Lower, L>;
            type Dim = Const<1>;

            const IS_REPEATABLE: bool = $repeatable;
            const SPLIT_MASK: usize = 1;

            unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
                let count = self.stride * index as isize;

                $view::new_unchecked(self.ptr.as_ptr().offset(count), self.mapping)
            }

            fn shape(&self) -> [usize; 1] {
                [self.size]
            }
        }
    };
}

impl_axis_expr!(AxisExpr, ViewArray, const, true);
impl_axis_expr!(AxisExprMut, ViewArrayMut, mut, false);

impl<'a, T, D: Dim, L: Layout> Clone for AxisExpr<'a, T, D, L> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            mapping: self.mapping,
            size: self.size,
            stride: self.stride,
            phantom: PhantomData,
        }
    }
}

unsafe impl<'a, T: Sync, D: Dim, L: Layout> Send for AxisExpr<'a, T, D, L> {}
unsafe impl<'a, T: Sync, D: Dim, L: Layout> Sync for AxisExpr<'a, T, D, L> {}

unsafe impl<'a, T: Send, D: Dim, L: Layout> Send for AxisExprMut<'a, T, D, L> {}
unsafe impl<'a, T: Sync, D: Dim, L: Layout> Sync for AxisExprMut<'a, T, D, L> {}

impl<'a, T, D: Dim, A: Allocator> Drain<'a, T, D, A> {
    pub(crate) fn new(grid: &'a mut GridArray<T, D, A>, start: usize, end: usize) -> Self {
        assert!(start <= end && end <= grid.size(D::RANK - 1), "invalid range");

        let tail = grid.size(D::RANK - 1) - end;
        let inner_len = grid.shape()[..D::RANK - 1].iter().product::<usize>();
        let index = start * inner_len;

        // Shrink the array, to be safe in case Drain is leaked.
        unsafe {
            grid.buffer.set_mapping(Mapping::resize_dim(grid.mapping(), D::RANK - 1, start));
        }

        Self { grid, start, end, tail, inner_len, index }
    }
}

impl<'a, T: Debug, D: Dim, A: Allocator> Debug for Drain<'a, T, D, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        assert!(self.index == self.start * self.inner_len, "expression in use");

        let ptr = unsafe { self.grid.as_ptr().add(self.index) };
        let mapping = Mapping::resize_dim(self.grid.mapping(), D::RANK - 1, self.end - self.start);
        let view = unsafe { ViewArray::<T, D, Dense>::new_unchecked(ptr, mapping) };

        f.debug_tuple("Drain").field(&view).finish()
    }
}

impl<'a, T, D: Dim, A: Allocator> Drop for Drain<'a, T, D, A> {
    fn drop(&mut self) {
        struct DropGuard<'a, 'b, T, D: Dim, A: Allocator>(&'b mut Drain<'a, T, D, A>);

        impl<'a, 'b, T, D: Dim, A: Allocator> Drop for DropGuard<'a, 'b, T, D, A> {
            fn drop(&mut self) {
                let size = self.0.start + self.0.tail;
                let mapping = Mapping::resize_dim(self.0.grid.mapping(), D::RANK - 1, size);

                unsafe {
                    let src = self.0.grid.as_ptr().add(self.0.end * self.0.inner_len);
                    let dst = self.0.grid.as_mut_ptr().add(self.0.start * self.0.inner_len);

                    ptr::copy(src, dst, self.0.tail * self.0.inner_len);
                    self.0.grid.buffer.set_mapping(mapping);
                }
            }
        }

        let guard = DropGuard(self);

        unsafe {
            let ptr = guard.0.grid.as_mut_ptr().add(guard.0.index);
            let len = guard.0.end * guard.0.inner_len - guard.0.index;

            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}

impl<'a, T, D: Dim, A: Allocator> Producer for Drain<'a, T, D, A> {
    type Item = T;
    type Dim = D;

    const IS_REPEATABLE: bool = false;
    const SPLIT_MASK: usize = (1 << D::RANK) >> 1;

    fn shape(&self) -> D::Shape {
        Mapping::resize_dim(self.grid.mapping(), D::RANK - 1, self.end - self.start).shape()
    }

    unsafe fn get_unchecked(&mut self, _: usize) -> Self::Item {
        debug_assert!(self.index < self.end * self.inner_len, "index out of bounds");

        self.index += 1; // Keep track of that the element is moved out.

        self.grid.as_mut_ptr().add(self.index - 1).read()
    }
}

unsafe impl<'a, T: Send, D: Dim, A: Allocator> Send for Drain<'a, T, D, A> {}
unsafe impl<'a, T: Sync, D: Dim, A: Allocator> Sync for Drain<'a, T, D, A> {}

impl<T> Fill<T> {
    pub(crate) fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Debug> Debug for Fill<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Fill").field(&self.value).finish()
    }
}

impl<T: Clone> Producer for Fill<T> {
    type Item = T;
    type Dim = Const<0>;

    const IS_REPEATABLE: bool = true;
    const SPLIT_MASK: usize = 0;

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        self.value.clone()
    }

    fn shape(&self) -> [usize; 0] {
        []
    }
}

impl<F> FillWith<F> {
    pub(crate) fn new(f: F) -> Self {
        Self { f }
    }
}

impl<T: Debug, F: FnMut() -> T> Debug for FillWith<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("FillWith").finish()
    }
}

impl<T, F: FnMut() -> T> Producer for FillWith<F> {
    type Item = T;
    type Dim = Const<0>;

    const IS_REPEATABLE: bool = true;
    const SPLIT_MASK: usize = 0;

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        (self.f)()
    }

    fn shape(&self) -> [usize; 0] {
        []
    }
}

impl<S: Shape, T> FromElem<S, T> {
    pub(crate) fn new(shape: S, elem: T) -> Self {
        _ = <S::Dim as Dim>::checked_len(shape);

        Self { shape, elem }
    }
}

impl<S: Shape, T: Debug> Debug for FromElem<S, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("FromElem").field(&self.shape).field(&self.elem).finish()
    }
}

impl<S: Shape, T: Clone> Producer for FromElem<S, T> {
    type Item = T;
    type Dim = S::Dim;

    const IS_REPEATABLE: bool = true;
    const SPLIT_MASK: usize = 0;

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        self.elem.clone()
    }

    fn shape(&self) -> S {
        self.shape
    }
}

impl<S: Shape, F> FromFn<S, F> {
    pub(crate) fn new(shape: S, f: F) -> Self {
        _ = <S::Dim as Dim>::checked_len(shape);

        Self { shape, f, index: S::default() }
    }
}

impl<S: Shape, T: Debug, F: FnMut(S) -> T> Debug for FromFn<S, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("FromFn").field(&self.shape).finish()
    }
}

impl<S: Shape, T, F: FnMut(S) -> T> Producer for FromFn<S, F> {
    type Item = T;
    type Dim = S::Dim;

    const IS_REPEATABLE: bool = true;
    const SPLIT_MASK: usize = (1 << S::Dim::RANK) - 1;

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        if S::Dim::RANK > 0 {
            self.index[0] = index;
        }

        (self.f)(self.index)
    }

    unsafe fn reset_dim(&mut self, dim: usize, _: usize) {
        self.index[dim] = 0;
    }

    fn shape(&self) -> S {
        self.shape
    }

    unsafe fn step_dim(&mut self, dim: usize) {
        self.index[dim] += 1;
    }
}

impl<T, D: Dim, A: Allocator> IntoExpr<T, D, A> {
    pub(crate) fn new(grid: GridArray<T, D, A>) -> Self {
        Self { grid, index: 0 }
    }
}

impl<T: Clone, D: Dim, A: Allocator + Clone> Clone for IntoExpr<T, D, A> {
    fn clone(&self) -> Self {
        assert!(self.index == 0, "expression in use");

        Self { grid: self.grid.clone(), index: self.index }
    }

    fn clone_from(&mut self, source: &Self) {
        assert!(self.index == 0 && source.index == 0, "expression in use");

        self.grid.clone_from(&source.grid);
    }
}

impl<T: Debug, D: Dim, A: Allocator> Debug for IntoExpr<T, D, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        assert!(self.index == 0, "expression in use");

        f.debug_tuple("IntoExpr").field(&self.grid).finish()
    }
}

impl<T, D: Dim, A: Allocator> Drop for IntoExpr<T, D, A> {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.grid.as_mut_ptr().add(self.index);
            let len = self.grid.len() - self.index;

            self.grid.buffer.set_mapping(Default::default());
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(ptr, len));
        }
    }
}

impl<T, D: Dim, A: Allocator> Producer for IntoExpr<T, D, A> {
    type Item = T;
    type Dim = D;

    const IS_REPEATABLE: bool = false;
    const SPLIT_MASK: usize = (1 << D::RANK) >> 1;

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        debug_assert!(self.index < self.grid.len(), "index out of bounds");

        self.index += 1; // Keep track of that the element is moved out.

        self.grid.as_mut_ptr().add(self.index - 1).read()
    }

    fn shape(&self) -> D::Shape {
        self.grid.shape()
    }
}

unsafe impl<T: Send, D: Dim, A: Allocator> Send for IntoExpr<T, D, A> {}
unsafe impl<T: Sync, D: Dim, A: Allocator> Sync for IntoExpr<T, D, A> {}

macro_rules! impl_view_expr {
    ($name:tt, $view:tt, $raw_mut:tt, {$($mut:tt)?}, $repeatable:tt) => {
        impl<'a, T, D: Dim, L: Layout> $name<'a, T, D, L> {
            pub(crate) unsafe fn new_unchecked(ptr: *$raw_mut T, mapping: L::Mapping<D>) -> Self {
                Self { ptr: NonNull::new_unchecked(ptr as *mut T), mapping, phantom: PhantomData }
            }
        }

        impl<'a, T: Debug, D: Dim, L: Layout> Debug for $name<'a, T, D, L> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let view = unsafe { // Assuming expression not in use.
                    ViewArray::<T, D, L>::new_unchecked(self.ptr.as_ptr(), self.mapping)
                };

                f.debug_tuple(stringify!($name)).field(&view).finish()
            }
        }

        impl<'a, T, D: Dim, L: Layout> Producer for $name<'a, T, D, L> {
            type Item = &'a $($mut)? T;
            type Dim = D;

            const IS_REPEATABLE: bool = $repeatable;
            const SPLIT_MASK: usize =
                if L::IS_UNIFORM { (1 << D::RANK) >> 1 } else { (1 << D::RANK) - 1 };

            unsafe fn get_unchecked(&mut self, index: usize) -> &'a $($mut)? T {
                let count = if D::RANK > 0 { self.mapping.stride(0) * index as isize } else { 0 };

                &$($mut)? *self.ptr.as_ptr().offset(count)
            }

            unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
                let count = -self.mapping.stride(dim) * count as isize;

                self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().offset(count));
            }

            fn shape(&self) -> D::Shape {
                self.mapping.shape()
            }

            unsafe fn step_dim(&mut self, dim: usize) {
                let count = self.mapping.stride(dim);

                self.ptr = NonNull::new_unchecked(self.ptr.as_ptr().offset(count));
            }
        }
    }
}

impl_view_expr!(Expr, ViewArray, const, {}, true);
impl_view_expr!(ExprMut, ViewArrayMut, mut, {mut}, false);

impl<'a, T, D: Dim, L: Layout> Clone for Expr<'a, T, D, L> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr, mapping: self.mapping, phantom: PhantomData }
    }
}

unsafe impl<'a, T: Sync, D: Dim, L: Layout> Send for Expr<'a, T, D, L> {}
unsafe impl<'a, T: Sync, D: Dim, L: Layout> Sync for Expr<'a, T, D, L> {}

unsafe impl<'a, T: Send, D: Dim, L: Layout> Send for ExprMut<'a, T, D, L> {}
unsafe impl<'a, T: Sync, D: Dim, L: Layout> Sync for ExprMut<'a, T, D, L> {}
