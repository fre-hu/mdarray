use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr;

use crate::buffer::Buffer;
use crate::dim::Const;
use crate::expr::{Expr, ExprMut, IntoExpr, Map, Zip};
use crate::expression::Expression;
use crate::grid::Grid;
use crate::index::SpanIndex;
use crate::iter::Iter;
use crate::layout::{Dense, Layout};
use crate::shape::{ConstShape, IntoShape, Shape};
use crate::span::Span;
use crate::traits::{Apply, FromExpression, IntoExpression};

/// Multidimensional array with constant-sized dimensions and inline allocation.
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct Array<T, S: ConstShape>(pub S::Array<T>);

impl<T, S: ConstShape> Array<T, S> {
    /// Converts an array with a single element into the contained value.
    ///
    /// # Panics
    ///
    /// Panics if the array length is not equal to one.
    pub fn into_scalar(self) -> T {
        assert!(self.len() == 1, "invalid length");

        self.into_shape(()).0
    }

    /// Converts the array into a reshaped array, which must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed.
    pub fn into_shape<I>(self, shape: I) -> Array<T, I::IntoShape>
    where
        I: IntoShape<IntoShape: ConstShape>,
    {
        assert!(shape.into_shape().len() == self.len(), "length must not change");

        let me = ManuallyDrop::new(self);

        unsafe { mem::transmute_copy(&me) }
    }
}

impl<'a, T, U, S: ConstShape> Apply<U> for &'a Array<T, S> {
    type Output<F: FnMut(&'a T) -> U> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a T, I::Item)) -> U> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    fn apply<F: FnMut(&'a T) -> U>(self, f: F) -> Self::Output<F> {
        self.expr().map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a T, I::Item)) -> U,
    {
        self.expr().zip(expr).map(f)
    }
}

impl<'a, T, U, S: ConstShape> Apply<U> for &'a mut Array<T, S> {
    type Output<F: FnMut(&'a mut T) -> U> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a mut T, I::Item)) -> U> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    fn apply<F: FnMut(&'a mut T) -> U>(self, f: F) -> Self::Output<F> {
        self.expr_mut().map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a mut T, I::Item)) -> U,
    {
        self.expr_mut().zip(expr).map(f)
    }
}

impl<T, U, S: ConstShape> Apply<U> for Array<T, S> {
    type Output<F: FnMut(T) -> U> = Array<U, S>;
    type ZippedWith<I: IntoExpression, F: FnMut((T, I::Item)) -> U> = Array<U, S>;

    fn apply<F: FnMut(T) -> U>(self, f: F) -> Array<U, S> {
        from_expr(self.into_expr().map(f))
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Array<U, S>
    where
        F: FnMut((T, I::Item)) -> U,
    {
        from_expr(self.into_expr().zip(expr).map(f))
    }
}

impl<T, U: ?Sized, S: ConstShape> AsMut<U> for Array<T, S>
where
    Span<T, S>: AsMut<U>,
{
    fn as_mut(&mut self) -> &mut U {
        (**self).as_mut()
    }
}

impl<T, U: ?Sized, S: ConstShape> AsRef<U> for Array<T, S>
where
    Span<T, S>: AsRef<U>,
{
    fn as_ref(&self) -> &U {
        (**self).as_ref()
    }
}

impl<T, S: ConstShape> Borrow<Span<T, S>> for Array<T, S> {
    fn borrow(&self) -> &Span<T, S> {
        self
    }
}

impl<T, S: ConstShape> BorrowMut<Span<T, S>> for Array<T, S> {
    fn borrow_mut(&mut self) -> &mut Span<T, S> {
        self
    }
}

impl<T: Debug, S: ConstShape> Debug for Array<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        (**self).fmt(f)
    }
}

impl<T, S: ConstShape> Deref for Array<T, S> {
    type Target = Span<T, S>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*(self as *const Array<T, S> as *const Span<T, S>) }
    }
}

impl<T, S: ConstShape> DerefMut for Array<T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self as *mut Array<T, S> as *mut Span<T, S>) }
    }
}

impl<T, S: ConstShape> From<Grid<T, S>> for Array<T, S> {
    fn from(value: Grid<T, S>) -> Self {
        Self::from_expr(value)
    }
}

impl<'a, T: 'a + Clone, S: ConstShape, L: Layout, I> From<I> for Array<T, S>
where
    I: IntoExpression<IntoExpr = Expr<'a, T, S, L>>,
{
    fn from(value: I) -> Self {
        Self::from_expr(value.into_expr().cloned())
    }
}

impl<B: Buffer<Shape: ConstShape>> From<IntoExpr<B>> for Array<B::Item, B::Shape> {
    fn from(value: IntoExpr<B>) -> Self {
        Self::from_expr(value)
    }
}

macro_rules! impl_from_array {
    ($n:tt, ($($xyz:tt),+), $array:tt) => {
        #[allow(unused_parens)]
        impl<T: Clone $(,const $xyz: usize)+> From<&$array> for Array<T, ($(Const<$xyz>),+)> {
            fn from(array: &$array) -> Self {
                Self(array.clone())
            }
        }

        #[allow(unused_parens)]
        impl<T $(,const $xyz: usize)+> From<Array<T, ($(Const<$xyz>),+)>> for $array {
            fn from(array: Array<T, ($(Const<$xyz>),*)>) -> Self {
                array.0
            }
        }

        #[allow(unused_parens)]
        impl<T $(,const $xyz: usize)+> From<$array> for Array<T, ($(Const<$xyz>),+)> {
            fn from(array: $array) -> Self {
                Self(array)
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

impl<T, S: ConstShape> FromExpression<T, S> for Array<T, S> {
    type WithConst<const N: usize> = S::WithConst<T, N, Self>;

    fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self {
        from_expr(expr.into_expr())
    }
}

impl<T: Hash, S: ConstShape> Hash for Array<T, S> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T, S: ConstShape, I: SpanIndex<T, S, Dense>> Index<I> for Array<T, S> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, S: ConstShape, I: SpanIndex<T, S, Dense>> IndexMut<I> for Array<T, S> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: ConstShape> IntoExpression for &'a Array<T, S> {
    type Shape = S;
    type IntoExpr = Expr<'a, T, S>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, T, S: ConstShape> IntoExpression for &'a mut Array<T, S> {
    type Shape = S;
    type IntoExpr = ExprMut<'a, T, S>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<T, S: ConstShape> IntoExpression for Array<T, S> {
    type Shape = S;
    type IntoExpr = IntoExpr<Array<ManuallyDrop<T>, S>>;

    fn into_expr(self) -> Self::IntoExpr {
        let me = ManuallyDrop::new(self);

        unsafe { IntoExpr::new(mem::transmute_copy(&me)) }
    }
}

impl<'a, T, S: ConstShape> IntoIterator for &'a Array<T, S> {
    type Item = &'a T;
    type IntoIter = Iter<Expr<'a, T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S: ConstShape> IntoIterator for &'a mut Array<T, S> {
    type Item = &'a mut T;
    type IntoIter = Iter<ExprMut<'a, T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, S: ConstShape> IntoIterator for Array<T, S> {
    type Item = T;
    type IntoIter = Iter<IntoExpr<Array<ManuallyDrop<T>, S>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_expr().into_iter()
    }
}

fn from_expr<T, S: ConstShape, E: Expression<Item = T>>(expr: E) -> Array<T, S> {
    struct DropGuard<'a, T, S: ConstShape> {
        array: &'a mut MaybeUninit<Array<T, S>>,
        index: usize,
    }

    impl<'a, T, S: ConstShape> Drop for DropGuard<'a, T, S> {
        fn drop(&mut self) {
            let ptr = self.array.as_mut_ptr() as *mut T;

            unsafe {
                ptr::slice_from_raw_parts_mut(ptr, self.index).drop_in_place();
            }
        }
    }

    assert!(expr.dims()[..] == S::default().dims()[..], "invalid shape");

    let mut array = MaybeUninit::uninit();
    let mut guard = DropGuard { array: &mut array, index: 0 };

    let ptr = guard.array.as_mut_ptr() as *mut T;

    expr.for_each(|x| unsafe {
        ptr.add(guard.index).write(x);
        guard.index += 1;
    });

    mem::forget(guard);

    unsafe { array.assume_init() }
}
