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
use crate::index::SpanIndex;
use crate::layout::{Dense, Layout, Uniform};

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
                _ = list.entries(self.as_span().outer_iter());
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

        if !self.as_span().is_empty() {
            hash(self.as_span(), state);
        }
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

impl<'a, L: 'a + Uniform, B: Buffer<Layout = L> + ?Sized> IntoIterator for &'a Array<B> {
    type Item = &'a B::Item;
    type IntoIter = L::Iter<'a, B::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_span().iter()
    }
}

impl<'a, L: 'a + Uniform, B: BufferMut<Layout = L> + ?Sized> IntoIterator for &'a mut Array<B> {
    type Item = &'a mut B::Item;
    type IntoIter = L::IterMut<'a, B::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_span().iter_mut()
    }
}

fn hash<T: Hash, L: Layout>(this: &SpanArray<T, impl Dim, L>, state: &mut impl Hasher) {
    if L::IS_UNIFORM {
        for x in this.flatten().iter() {
            x.hash(state);
        }
    } else {
        for x in this.outer_iter() {
            hash(&x, state);
        }
    }
}
