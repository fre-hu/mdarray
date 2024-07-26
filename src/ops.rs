#[cfg(feature = "nightly")]
use std::alloc::Allocator;

use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::dim::{Dim, Shape};
use crate::expr::{Drain, Expr, ExprMut, Fill, FillWith, FromElem, FromFn, IntoExpr, Map};
use crate::expression::Expression;
use crate::grid::Grid;
use crate::layout::Layout;
use crate::span::Span;
use crate::traits::{Apply, IntoExpression};

/// Range constructed from a unit spaced range with the given step size.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct StepRange<R, S> {
    /// Unit spaced range.
    pub range: R,

    /// Step size.
    pub step: S,
}

/// Creates a range with the given step size from a unit spaced range.
///
/// If the step size is negative, the result is obtained by reversing the input range
/// and stepping by the absolute value of the step size.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, step};
///
/// let v = expr![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///
/// assert_eq!(v.view(step(0..10, 2)).to_vec(), [0, 2, 4, 6, 8]);
/// assert_eq!(v.view(step(0..10, -2)).to_vec(), [9, 7, 5, 3, 1]);
/// ```
pub fn step<R, S>(range: R, step: S) -> StepRange<R, S> {
    StepRange { range, step }
}

impl<T: Eq, D: Dim, L: Layout> Eq for Expr<'_, T, D, L> {}
impl<T: Eq, D: Dim, L: Layout> Eq for ExprMut<'_, T, D, L> {}
impl<T: Eq, D: Dim, A: Allocator> Eq for Grid<T, D, A> {}
impl<T: Eq, D: Dim, L: Layout> Eq for Span<T, D, L> {}

impl<T, U, D: Dim, L: Layout, M: Layout, I: ?Sized> PartialEq<I> for Expr<'_, T, D, L>
where
    for<'a> &'a I: IntoExpression<IntoExpr = Expr<'a, U, D, M>>,
    T: PartialEq<U>,
{
    fn eq(&self, other: &I) -> bool {
        eq(self, &other.into_expr())
    }
}

impl<T, U, D: Dim, L: Layout, M: Layout, I: ?Sized> PartialEq<I> for ExprMut<'_, T, D, L>
where
    for<'a> &'a I: IntoExpression<IntoExpr = Expr<'a, U, D, M>>,
    T: PartialEq<U>,
{
    fn eq(&self, other: &I) -> bool {
        eq(self, &other.into_expr())
    }
}

impl<T, U, D: Dim, L: Layout, A: Allocator, I: ?Sized> PartialEq<I> for Grid<T, D, A>
where
    for<'a> &'a I: IntoExpression<IntoExpr = Expr<'a, U, D, L>>,
    T: PartialEq<U>,
{
    fn eq(&self, other: &I) -> bool {
        eq(self, &other.into_expr())
    }
}

impl<T, U, D: Dim, L: Layout, M: Layout, I: ?Sized> PartialEq<I> for Span<T, D, L>
where
    for<'a> &'a I: IntoExpression<IntoExpr = Expr<'a, U, D, M>>,
    T: PartialEq<U>,
{
    fn eq(&self, other: &I) -> bool {
        eq(self, &other.into_expr())
    }
}

macro_rules! impl_binary_op {
    ($trt:tt, $fn:tt) => {
        impl<'a, T, U, D: Dim, L: Layout, I: Apply<U>> $trt<I> for &'a Expr<'_, T, D, L>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, D: Dim, L: Layout, I: Apply<U>> $trt<I> for &'a ExprMut<'_, T, D, L>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, D: Dim, A: Allocator, I: Apply<U>> $trt<I> for &'a Grid<T, D, A>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, D: Dim, L: Layout, I: Apply<U>> $trt<I> for &'a Span<T, D, L>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, D: Dim, A: Allocator, I: Apply<U>> $trt<I> for Drain<'a, T, D, A>
        where
            T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<'a, T, U, D: Dim, L: Layout, I: Apply<U>> $trt<I> for Expr<'a, T, D, L>
        where
            &'a T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, &'a T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T: Clone, U, I: Apply<U>> $trt<I> for Fill<T>
        where
            T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T: Clone, U, F: FnMut() -> T, I: Apply<U>> $trt<I> for FillWith<F>
        where
            T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<S: Shape, T: Clone, U, I: Apply<U>> $trt<I> for FromElem<S, T>
        where
            T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<S: Shape, T, U, F: FnMut(S) -> T, I: Apply<U>> $trt<I> for FromFn<S, F>
        where
            T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T, D: Dim, I: IntoExpression, A: Allocator> $trt<I> for Grid<T, D, A>
        where
            T: $trt<I::Item, Output = T>,
        {
            type Output = Self;

            fn $fn(self, rhs: I) -> Self {
                self.zip_with(rhs, |(x, y)| x.$fn(y))
            }
        }

        impl<T, U, D: Dim, A: Allocator, I: Apply<U>> $trt<I> for IntoExpr<T, D, A>
        where
            T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }

        impl<T, U, E: Expression, F: FnMut(E::Item) -> T, I: Apply<U>> $trt<I> for Map<E, F>
        where
            T: $trt<I::Item, Output = U>,
        {
            type Output = I::ZippedWith<Self, impl FnMut((I::Item, T)) -> U>;

            fn $fn(self, rhs: I) -> Self::Output {
                rhs.zip_with(self, |(x, y)| y.$fn(x))
            }
        }
    };
}

impl_binary_op!(Add, add);
impl_binary_op!(Sub, sub);
impl_binary_op!(Mul, mul);
impl_binary_op!(Div, div);
impl_binary_op!(Rem, rem);
impl_binary_op!(BitAnd, bitand);
impl_binary_op!(BitOr, bitor);
impl_binary_op!(BitXor, bitxor);
impl_binary_op!(Shl, shl);
impl_binary_op!(Shr, shr);

macro_rules! impl_op_assign {
    ($trt:tt, $fn:tt) => {
        impl<T, D: Dim, L: Layout, I: IntoExpression> $trt<I> for ExprMut<'_, T, D, L>
        where
            T: $trt<I::Item>,
        {
            fn $fn(&mut self, rhs: I) {
                self.expr_mut().zip(rhs).for_each(|(x, y)| x.$fn(y));
            }
        }

        impl<T, D: Dim, I: IntoExpression, A: Allocator> $trt<I> for Grid<T, D, A>
        where
            T: $trt<I::Item>,
        {
            fn $fn(&mut self, rhs: I) {
                self.expr_mut().zip(rhs).for_each(|(x, y)| x.$fn(y));
            }
        }

        impl<T, D: Dim, L: Layout, I: IntoExpression> $trt<I> for Span<T, D, L>
        where
            T: $trt<I::Item>,
        {
            fn $fn(&mut self, rhs: I) {
                self.expr_mut().zip(rhs).for_each(|(x, y)| x.$fn(y));
            }
        }
    };
}

impl_op_assign!(AddAssign, add_assign);
impl_op_assign!(SubAssign, sub_assign);
impl_op_assign!(MulAssign, mul_assign);
impl_op_assign!(DivAssign, div_assign);
impl_op_assign!(RemAssign, rem_assign);
impl_op_assign!(BitAndAssign, bitand_assign);
impl_op_assign!(BitOrAssign, bitor_assign);
impl_op_assign!(BitXorAssign, bitxor_assign);
impl_op_assign!(ShlAssign, shl_assign);
impl_op_assign!(ShrAssign, shr_assign);

macro_rules! impl_unary_op {
    ($trt:tt, $fn:tt) => {
        impl<'a, T, U, D: Dim, L: Layout> $trt for &'a Expr<'_, T, D, L>
        where
            &'a T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, D: Dim, L: Layout> $trt for &'a ExprMut<'_, T, D, L>
        where
            &'a T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, D: Dim, A: Allocator> $trt for &'a Grid<T, D, A>
        where
            &'a T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, D: Dim, L: Layout> $trt for &'a Span<T, D, L>
        where
            &'a T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, D: Dim, A: Allocator> $trt for Drain<'a, T, D, A>
        where
            T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<'a, T, U, D: Dim, L: Layout> $trt for Expr<'a, T, D, L>
        where
            &'a T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(&'a T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T: Clone, U> $trt for Fill<T>
        where
            T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T: Clone, U, F: FnMut() -> T> $trt for FillWith<F>
        where
            T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<S: Shape, T: Clone, U> $trt for FromElem<S, T>
        where
            T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<S: Shape, T, U, F: FnMut(S) -> T> $trt for FromFn<S, F>
        where
            T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T, D: Dim, A: Allocator> $trt for Grid<T, D, A>
        where
            T: $trt<Output = T>,
        {
            type Output = Self;

            fn $fn(self) -> Self {
                self.apply(|x| x.$fn())
            }
        }

        impl<T, U, D: Dim, A: Allocator> $trt for IntoExpr<T, D, A>
        where
            T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }

        impl<T, U, E: Expression, F: FnMut(E::Item) -> T> $trt for Map<E, F>
        where
            T: $trt<Output = U>,
        {
            type Output = <Self as Apply<U>>::Output<impl FnMut(T) -> U>;

            fn $fn(self) -> Self::Output {
                self.apply(|x| x.$fn())
            }
        }
    };
}

impl_unary_op!(Neg, neg);
impl_unary_op!(Not, not);

fn eq<T, U, D: Dim, L: Layout, M: Layout>(this: &Span<T, D, L>, other: &Span<U, D, M>) -> bool
where
    T: PartialEq<U>,
{
    if this.shape()[..] == other.shape()[..] {
        if L::IS_UNIFORM && M::IS_UNIFORM {
            if L::IS_UNIT_STRIDED && M::IS_UNIT_STRIDED {
                this.remap()[..].eq(&other.remap()[..])
            } else {
                this.iter().eq(other)
            }
        } else {
            this.outer_expr().into_iter().eq(other.outer_expr())
        }
    } else {
        false
    }
}
