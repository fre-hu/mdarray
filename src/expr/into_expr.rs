use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr;

use crate::buffer::Buffer;
use crate::expr::adapters::{Map, Zip};
use crate::expr::expr::{Expr, ExprMut};
use crate::expression::Expression;
use crate::index::SpanIndex;
use crate::iter::Iter;
use crate::layout::Dense;
use crate::shape::Shape;
use crate::span::Span;
use crate::traits::{Apply, IntoExpression};

/// Expression that moves elements out of an array.
pub struct IntoExpr<B: Buffer> {
    buffer: B,
    index: usize,
}

impl<B: Buffer> IntoExpr<B> {
    pub(crate) fn new(buffer: B) -> Self {
        Self { buffer, index: 0 }
    }
}

impl<'a, T, B: Buffer> Apply<T> for &'a IntoExpr<B> {
    type Output<F: FnMut(&'a B::Item) -> T> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a B::Item, I::Item)) -> T> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    fn apply<F: FnMut(&'a B::Item) -> T>(self, f: F) -> Self::Output<F> {
        self.expr().map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a B::Item, I::Item)) -> T,
    {
        self.expr().zip(expr).map(f)
    }
}

impl<'a, T, B: Buffer> Apply<T> for &'a mut IntoExpr<B> {
    type Output<F: FnMut(&'a mut B::Item) -> T> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a mut B::Item, I::Item)) -> T> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    fn apply<F: FnMut(&'a mut B::Item) -> T>(self, f: F) -> Self::Output<F> {
        self.expr_mut().map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a mut B::Item, I::Item)) -> T,
    {
        self.expr_mut().zip(expr).map(f)
    }
}

impl<T: ?Sized, B: Buffer> AsMut<T> for IntoExpr<B>
where
    Span<B::Item, B::Shape>: AsMut<T>,
{
    fn as_mut(&mut self) -> &mut T {
        (**self).as_mut()
    }
}

impl<T: ?Sized, B: Buffer> AsRef<T> for IntoExpr<B>
where
    Span<B::Item, B::Shape>: AsRef<T>,
{
    fn as_ref(&self) -> &T {
        (**self).as_ref()
    }
}

impl<B: Buffer> Borrow<Span<B::Item, B::Shape>> for IntoExpr<B> {
    fn borrow(&self) -> &Span<B::Item, B::Shape> {
        self
    }
}

impl<B: Buffer> BorrowMut<Span<B::Item, B::Shape>> for IntoExpr<B> {
    fn borrow_mut(&mut self) -> &mut Span<B::Item, B::Shape> {
        self
    }
}

impl<B: Buffer + Clone> Clone for IntoExpr<B> {
    fn clone(&self) -> Self {
        assert!(self.index == 0, "expression in use");

        Self { buffer: self.buffer.clone(), index: 0 }
    }

    fn clone_from(&mut self, source: &Self) {
        assert!(self.index == 0 && source.index == 0, "expression in use");

        self.buffer.clone_from(&source.buffer);
    }
}

impl<B: Buffer<Item: Debug>> Debug for IntoExpr<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        (**self).fmt(f)
    }
}

impl<B: Buffer + Default> Default for IntoExpr<B> {
    fn default() -> Self {
        Self { buffer: Default::default(), index: 0 }
    }
}

impl<B: Buffer> Deref for IntoExpr<B> {
    type Target = Span<B::Item, B::Shape>;

    fn deref(&self) -> &Self::Target {
        debug_assert!(self.index == 0, "expression in use");

        let span = self.buffer.as_span();

        unsafe { &*(span as *const Span<ManuallyDrop<B::Item>, B::Shape> as *const Self::Target) }
    }
}

impl<B: Buffer> DerefMut for IntoExpr<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        debug_assert!(self.index == 0, "expression in use");

        let span = self.buffer.as_mut_span();

        unsafe { &mut *(span as *mut Span<ManuallyDrop<B::Item>, B::Shape> as *mut Self::Target) }
    }
}

impl<B: Buffer> Drop for IntoExpr<B> {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.buffer.as_mut_span().as_mut_ptr().add(self.index) as *mut B::Item;
            let len = self.buffer.as_span().len() - self.index;

            ptr::slice_from_raw_parts_mut(ptr, len).drop_in_place();
        }
    }
}

impl<B: Buffer> Expression for IntoExpr<B> {
    type Shape = B::Shape;

    const IS_REPEATABLE: bool = false;
    const SPLIT_MASK: usize = (1 << B::Shape::RANK) >> 1;

    fn shape(&self) -> Self::Shape {
        self.buffer.as_span().shape()
    }

    unsafe fn get_unchecked(&mut self, _: usize) -> B::Item {
        debug_assert!(self.index < self.buffer.as_span().len(), "index out of bounds");

        self.index += 1; // Keep track of that the element is moved out.

        ManuallyDrop::take(&mut *self.buffer.as_mut_span().as_mut_ptr().add(self.index - 1))
    }

    unsafe fn reset_dim(&mut self, _: usize, _: usize) {}
    unsafe fn step_dim(&mut self, _: usize) {}
}

impl<B: Buffer<Item: Hash>> Hash for IntoExpr<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<B: Buffer, I: SpanIndex<B::Item, B::Shape, Dense>> Index<I> for IntoExpr<B> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<B: Buffer, I: SpanIndex<B::Item, B::Shape, Dense>> IndexMut<I> for IntoExpr<B> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, B: Buffer> IntoExpression for &'a IntoExpr<B> {
    type Shape = B::Shape;
    type IntoExpr = Expr<'a, B::Item, B::Shape>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, B: Buffer> IntoExpression for &'a mut IntoExpr<B> {
    type Shape = B::Shape;
    type IntoExpr = ExprMut<'a, B::Item, B::Shape>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<'a, B: Buffer> IntoIterator for &'a IntoExpr<B> {
    type Item = &'a B::Item;
    type IntoIter = Iter<Expr<'a, B::Item, B::Shape>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, B: Buffer> IntoIterator for &'a mut IntoExpr<B> {
    type Item = &'a mut B::Item;
    type IntoIter = Iter<ExprMut<'a, B::Item, B::Shape>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<B: Buffer> IntoIterator for IntoExpr<B> {
    type Item = B::Item;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Iter<Self> {
        Iter::new(self)
    }
}
