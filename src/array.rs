#[cfg(feature = "nightly")]
use std::alloc::Global;
use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::ops::{Index, IndexMut};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Global;
use crate::buffer::{Buffer, BufferMut, SizedBuffer, SizedBufferMut};
use crate::buffer::{GridBuffer, SpanBuffer, ViewBuffer, ViewBufferMut};
use crate::dim::Dim;
use crate::expr::{Expr, ExprMut};
use crate::expression::Expression;
use crate::index::SpanIndex;
use crate::iter::Iter;
use crate::layout::Dense;
use crate::traits::{Apply, IntoExpression};

/// Multidimensional array type with static rank.
#[repr(transparent)]
pub struct Array<B: ?Sized> {
    pub(crate) buffer: B,
}

pub(crate) type GridArray<T, D, A = Global> = Array<GridBuffer<T, D, A>>;
pub(crate) type SpanArray<T, D, L> = Array<SpanBuffer<T, D, L>>;
pub(crate) type ViewArray<'a, T, D, L> = Array<ViewBuffer<'a, T, D, L>>;
pub(crate) type ViewArrayMut<'a, T, D, L> = Array<ViewBufferMut<'a, T, D, L>>;

impl<B: Buffer + ?Sized> Array<B> {
    /// Returns an array span of the entire array.
    pub fn as_span(&self) -> &SpanArray<B::Item, B::Dim, B::Layout> {
        self.buffer.as_span()
    }
}

impl<B: BufferMut + ?Sized> Array<B> {
    /// Returns a mutable array span of the entire array.
    pub fn as_mut_span(&mut self) -> &mut SpanArray<B::Item, B::Dim, B::Layout> {
        self.buffer.as_mut_span()
    }
}

impl<'a, T, B: Buffer + ?Sized> Apply<T> for &'a Array<B> {
    type Output = GridArray<T, B::Dim>;
    type ZippedWith<I: IntoExpression> = GridArray<T, <B::Dim as Dim>::Max<I::Dim>>;

    fn apply<F: FnMut(Self::Item) -> T>(self, f: F) -> Self::Output {
        self.as_span().expr().map(f).eval()
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, mut f: F) -> Self::ZippedWith<I>
    where
        F: FnMut(Self::Item, I::Item) -> T,
    {
        self.as_span().expr().zip(expr).map(|(x, y)| f(x, y)).eval()
    }
}

impl<'a, T, B: BufferMut + ?Sized> Apply<T> for &'a mut Array<B> {
    type Output = GridArray<T, B::Dim>;
    type ZippedWith<I: IntoExpression> = GridArray<T, <B::Dim as Dim>::Max<I::Dim>>;

    fn apply<F: FnMut(Self::Item) -> T>(self, f: F) -> Self::Output {
        self.as_mut_span().expr_mut().map(f).eval()
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, mut f: F) -> Self::ZippedWith<I>
    where
        F: FnMut(Self::Item, I::Item) -> T,
    {
        self.as_mut_span().expr_mut().zip(expr).map(|(x, y)| f(x, y)).eval()
    }
}

impl<B: BufferMut<Layout = Dense> + ?Sized> AsMut<[B::Item]> for Array<B> {
    fn as_mut(&mut self) -> &mut [B::Item] {
        self.as_mut_span().as_mut_slice()
    }
}

impl<B: BufferMut + ?Sized> AsMut<SpanArray<B::Item, B::Dim, B::Layout>> for Array<B> {
    fn as_mut(&mut self) -> &mut SpanArray<B::Item, B::Dim, B::Layout> {
        self.as_mut_span()
    }
}

impl<B: Buffer<Layout = Dense> + ?Sized> AsRef<[B::Item]> for Array<B> {
    fn as_ref(&self) -> &[B::Item] {
        self.as_span().as_slice()
    }
}

impl<B: Buffer + ?Sized> AsRef<SpanArray<B::Item, B::Dim, B::Layout>> for Array<B> {
    fn as_ref(&self) -> &SpanArray<B::Item, B::Dim, B::Layout> {
        self.as_span()
    }
}

impl<B: SizedBuffer> Borrow<SpanArray<B::Item, B::Dim, B::Layout>> for Array<B> {
    fn borrow(&self) -> &SpanArray<B::Item, B::Dim, B::Layout> {
        self.as_span()
    }
}

impl<B: SizedBufferMut> BorrowMut<SpanArray<B::Item, B::Dim, B::Layout>> for Array<B> {
    fn borrow_mut(&mut self) -> &mut SpanArray<B::Item, B::Dim, B::Layout> {
        self.as_mut_span()
    }
}

impl<B: SizedBuffer + Clone> Clone for Array<B> {
    fn clone(&self) -> Self {
        Self { buffer: self.buffer.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.buffer.clone_from(&source.buffer);
    }
}

impl<B: SizedBuffer + Copy> Copy for Array<B> {}

impl<T: Debug, B: Buffer<Item = T> + ?Sized> Debug for Array<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if B::Dim::RANK == 0 {
            self.as_span()[<B::Dim as Dim>::Shape::default()].fmt(f)
        } else {
            let mut list = f.debug_list();

            // Empty arrays should give an empty list.
            if !self.as_span().is_empty() {
                _ = list.entries(self.as_span().outer_expr());
            }

            list.finish()
        }
    }
}

impl<B: SizedBuffer> Deref for Array<B> {
    type Target = SpanArray<B::Item, B::Dim, B::Layout>;

    fn deref(&self) -> &Self::Target {
        self.as_span()
    }
}

impl<B: SizedBufferMut> DerefMut for Array<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_span()
    }
}

impl<T: Hash, B: Buffer<Item = T> + ?Sized> Hash for Array<B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in 0..B::Dim::RANK {
            #[cfg(not(feature = "nightly"))]
            state.write_usize(self.as_span().size(i));
            #[cfg(feature = "nightly")]
            state.write_length_prefix(self.as_span().size(i));
        }

        self.as_span().expr().for_each(|x| x.hash(state));
    }
}

impl<B: Buffer + ?Sized, I: SpanIndex<B::Item, B::Dim, B::Layout>> Index<I> for Array<B> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self.as_span())
    }
}

impl<B: BufferMut + ?Sized, I: SpanIndex<B::Item, B::Dim, B::Layout>> IndexMut<I> for Array<B> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self.as_mut_span())
    }
}

impl<'a, B: Buffer + ?Sized> IntoExpression for &'a Array<B> {
    type Item = &'a B::Item;
    type Dim = B::Dim;
    type Producer = Expr<'a, B::Item, B::Dim, B::Layout>;

    fn into_expr(self) -> Expression<Self::Producer> {
        self.as_span().expr()
    }
}

impl<'a, B: BufferMut + ?Sized> IntoExpression for &'a mut Array<B> {
    type Item = &'a mut B::Item;
    type Dim = B::Dim;
    type Producer = ExprMut<'a, B::Item, B::Dim, B::Layout>;

    fn into_expr(self) -> Expression<Self::Producer> {
        self.as_mut_span().expr_mut()
    }
}

impl<'a, B: Buffer + ?Sized> IntoIterator for &'a Array<B> {
    type Item = &'a B::Item;
    type IntoIter = Iter<Expr<'a, B::Item, B::Dim, B::Layout>>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_span().iter()
    }
}

impl<'a, B: BufferMut + ?Sized> IntoIterator for &'a mut Array<B> {
    type Item = &'a mut B::Item;
    type IntoIter = Iter<ExprMut<'a, B::Item, B::Dim, B::Layout>>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_span().iter_mut()
    }
}
