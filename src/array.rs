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
use crate::format::{Dense, Format, Uniform};
use crate::index::span::SpanIndex;

/// Multidimensional array type with static rank.
#[repr(transparent)]
pub struct Array<B: ?Sized> {
    pub(crate) buffer: B,
}

pub type GridArray<T, D, A = Global> = Array<GridBuffer<T, D, A>>;
pub type SpanArray<T, D, F> = Array<SpanBuffer<T, D, F>>;
pub type ViewArray<'a, T, D, F> = Array<ViewBuffer<'a, T, D, F>>;
pub type ViewArrayMut<'a, T, D, F> = Array<ViewBufferMut<'a, T, D, F>>;

impl<B: Buffer + ?Sized> Array<B> {
    /// Returns an array span of the entire array.
    #[must_use]
    pub fn as_span(&self) -> &SpanArray<B::Item, B::Dim, B::Format> {
        self.buffer.as_span()
    }
}

impl<B: BufferMut + ?Sized> Array<B> {
    /// Returns a mutable array span of the entire array.
    #[must_use]
    pub fn as_mut_span(&mut self) -> &mut SpanArray<B::Item, B::Dim, B::Format> {
        self.buffer.as_mut_span()
    }
}

impl<B: BufferMut<Format = Dense> + ?Sized> AsMut<[B::Item]> for Array<B> {
    fn as_mut(&mut self) -> &mut [B::Item] {
        self.as_mut_span().as_mut_slice()
    }
}

impl<B: BufferMut + ?Sized> AsMut<SpanArray<B::Item, B::Dim, B::Format>> for Array<B> {
    fn as_mut(&mut self) -> &mut SpanArray<B::Item, B::Dim, B::Format> {
        self.as_mut_span()
    }
}

impl<B: Buffer<Format = Dense> + ?Sized> AsRef<[B::Item]> for Array<B> {
    fn as_ref(&self) -> &[B::Item] {
        self.as_span().as_slice()
    }
}

impl<B: Buffer + ?Sized> AsRef<SpanArray<B::Item, B::Dim, B::Format>> for Array<B> {
    fn as_ref(&self) -> &SpanArray<B::Item, B::Dim, B::Format> {
        self.as_span()
    }
}

impl<B: Buffer<Format = Dense> + ?Sized> Borrow<[B::Item]> for Array<B> {
    fn borrow(&self) -> &[B::Item] {
        self.as_span().as_slice()
    }
}

impl<B: SizedBuffer> Borrow<SpanArray<B::Item, B::Dim, B::Format>> for Array<B> {
    fn borrow(&self) -> &SpanArray<B::Item, B::Dim, B::Format> {
        self.as_span()
    }
}

impl<B: BufferMut<Format = Dense> + ?Sized> BorrowMut<[B::Item]> for Array<B> {
    fn borrow_mut(&mut self) -> &mut [B::Item] {
        self.as_mut_span().as_mut_slice()
    }
}

impl<B: SizedBufferMut> BorrowMut<SpanArray<B::Item, B::Dim, B::Format>> for Array<B> {
    fn borrow_mut(&mut self) -> &mut SpanArray<B::Item, B::Dim, B::Format> {
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

            if !self.as_span().is_empty() {
                if B::Dim::RANK == 1 {
                    list.entries(self.as_span().flatten().iter());
                } else {
                    list.entries(self.as_span().outer_iter());
                }
            }

            list.finish()
        }
    }
}

impl<B: SizedBuffer> Deref for Array<B> {
    type Target = SpanArray<B::Item, B::Dim, B::Format>;

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
        let is_empty = self.as_span().is_empty();
        let shape = if is_empty { Default::default() } else { self.as_span().shape() };

        for i in 0..B::Dim::RANK {
            #[cfg(not(feature = "nightly"))]
            state.write_usize(shape[i]);
            #[cfg(feature = "nightly")]
            state.write_length_prefix(shape[i]);
        }

        if !is_empty {
            hash(self.as_span(), state);
        }
    }
}

impl<B: Buffer + ?Sized, I: SpanIndex<B::Item, B::Dim, B::Format>> Index<I> for Array<B> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self.as_span())
    }
}

impl<B: BufferMut + ?Sized, I: SpanIndex<B::Item, B::Dim, B::Format>> IndexMut<I> for Array<B> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self.as_mut_span())
    }
}

impl<'a, F: 'a + Uniform, B: Buffer<Format = F> + ?Sized> IntoIterator for &'a Array<B> {
    type Item = &'a B::Item;
    type IntoIter = F::Iter<'a, B::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_span().iter()
    }
}

impl<'a, F: 'a + Uniform, B: BufferMut<Format = F> + ?Sized> IntoIterator for &'a mut Array<B> {
    type Item = &'a mut B::Item;
    type IntoIter = F::IterMut<'a, B::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_span().iter_mut()
    }
}

fn hash<T: Hash, F: Format>(this: &SpanArray<T, impl Dim, F>, state: &mut impl Hasher) {
    if F::IS_UNIFORM {
        for x in this.flatten().iter() {
            x.hash(state);
        }
    } else {
        for x in this.outer_iter() {
            hash(&x, state);
        }
    }
}
