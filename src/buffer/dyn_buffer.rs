use alloc::alloc::{Allocator, Global, Layout};
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::marker::PhantomData;
use core::mem::{self, ManuallyDrop, MaybeUninit};
use core::ptr;

use crate::array::Array;
use crate::buffer::{Buffer, Owned, StaticBuffer};
use crate::dim::Const;
use crate::expr::{Expression, IntoExpr};
use crate::layout::Dense;
use crate::mapping::{DenseMapping, Mapping};
use crate::raw_slice::RawSlice;
use crate::shape::{ConstShape, Shape};
use crate::slice::Slice;

/// Array buffer type with dynamic allocation.
pub struct DynBuffer<T, S: Shape, A: Allocator = Global> {
    slice: RawSlice<T, S, Dense>,
    capacity: usize,
    alloc: A,
}

struct DropGuard<'a, T, A: Allocator> {
    ptr: *mut T,
    len: usize,
    phantom: PhantomData<&'a mut Vec<T, A>>,
}

impl<T, S: Shape, A: Allocator> DynBuffer<T, S, A> {
    #[inline]
    pub(crate) fn capacity(&self) -> usize {
        if mem::size_of::<T>() > 0 { self.capacity } else { usize::MAX }
    }

    #[inline]
    pub(crate) unsafe fn from_parts(vec: Vec<T, A>, shape: S) -> Self {
        debug_assert!(Some(vec.len()) == shape.checked_len(), "length mismatch");

        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_allocator();

        Self {
            slice: unsafe { RawSlice::new_unchecked(ptr, DenseMapping::new(shape)) },
            capacity,
            alloc,
        }
    }

    #[inline]
    pub(crate) fn into_parts(self) -> (Vec<T, A>, S) {
        let mut me = ManuallyDrop::new(self);

        let vec = unsafe {
            Vec::from_raw_parts_in(
                me.slice.as_mut_ptr(),
                me.slice.mapping().len(),
                me.capacity,
                ptr::read(&me.alloc),
            )
        };

        unsafe { (vec, ptr::read(me.slice.mapping().shape())) }
    }

    #[inline]
    pub(crate) fn resize_with<F: FnMut() -> T>(&mut self, new_dims: &[usize], mut f: F) {
        assert!(new_dims.len() == self.as_slice().rank(), "invalid rank");

        if !new_dims.is_empty() {
            let new_len = new_dims.iter().try_fold(1usize, |acc, &x| acc.checked_mul(x));
            let new_len = new_len.expect("invalid length");

            unsafe {
                self.with_mut_parts(|vec, old_shape| {
                    old_shape.with_dims(|old_dims| {
                        if new_len == 0 {
                            vec.clear();
                        } else if new_dims[1..] == old_dims[1..] {
                            vec.resize_with(new_len, &mut f);
                        } else {
                            let mut new_vec = Vec::with_capacity_in(new_len, *vec.allocator());

                            copy_dim::<T, S, A>(
                                &mut DropGuard::new(vec),
                                &mut new_vec,
                                old_dims,
                                new_dims,
                                &mut f,
                            );

                            *vec = new_vec;
                        }
                    });

                    old_shape.with_mut_dims(|dims| dims.copy_from_slice(new_dims));
                });
            }
        }
    }

    #[inline]
    pub(crate) unsafe fn set_shape(&mut self, new_shape: S) {
        debug_assert!(new_shape.checked_len().is_some(), "invalid length");
        debug_assert!(new_shape.len() <= self.capacity, "length exceeds capacity");

        unsafe {
            *self.slice.mapping_mut() = DenseMapping::new(new_shape);
        }
    }

    #[inline]
    pub(crate) unsafe fn with_mut_parts<U, F>(&mut self, f: F) -> U
    where
        F: FnOnce(&mut Vec<T, &A>, &mut S) -> U,
    {
        struct DropGuard<'a, T, S: Shape, A: Allocator> {
            slice: &'a mut RawSlice<T, S, Dense>,
            capacity: &'a mut usize,
            vec: ManuallyDrop<Vec<T, &'a A>>,
        }

        impl<T, S: Shape, A: Allocator> Drop for DropGuard<'_, T, S, A> {
            #[inline]
            fn drop(&mut self) {
                unsafe {
                    self.slice.set_ptr(self.vec.as_mut_ptr());
                    *self.capacity = self.vec.capacity();

                    // Cleanup in case of length mismatch (e.g. due to allocation failure)
                    if self.vec.len() != self.slice.mapping().len() {
                        assert!(S::default().len() == 0, "default length not zero");

                        *self.slice.mapping_mut() = DenseMapping::default();
                        ptr::drop_in_place(self.vec.as_mut_slice());
                    }
                }
            }
        }

        let vec = unsafe {
            Vec::from_raw_parts_in(
                self.slice.as_mut_ptr(),
                self.slice.mapping().len(),
                self.capacity,
                &self.alloc,
            )
        };

        let mut guard = DropGuard {
            slice: &mut self.slice,
            capacity: &mut self.capacity,
            vec: ManuallyDrop::new(vec),
        };

        let shape = unsafe { guard.slice.mapping_mut().shape_mut() };
        let result = f(&mut guard.vec, shape);

        debug_assert!(Some(guard.vec.len()) == shape.checked_len(), "length mismatch");

        unsafe {
            guard.slice.set_ptr(guard.vec.as_mut_ptr());
            *guard.capacity = guard.vec.capacity();
        }

        mem::forget(guard);

        result
    }
}

impl<T, S: Shape, A: Allocator> Buffer for DynBuffer<T, S, A> {
    type Item = T;
    type Shape = S;

    #[inline]
    fn as_mut_slice(&mut self) -> &mut Slice<T, S> {
        self.slice.as_mut_slice()
    }

    #[inline]
    fn as_slice(&self) -> &Slice<T, S> {
        self.slice.as_slice()
    }
}

impl<T: Clone, S: Shape, A: Allocator + Clone> Clone for DynBuffer<T, S, A> {
    #[inline]
    fn clone(&self) -> Self {
        Owned::clone(self)
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        Owned::clone_from(self, source);
    }
}

impl<T, S: Shape, A: Allocator> Drop for DynBuffer<T, S, A> {
    #[inline]
    fn drop(&mut self) {
        _ = unsafe {
            Vec::from_raw_parts_in(
                self.slice.as_mut_ptr(),
                self.slice.mapping().len(),
                self.capacity,
                &self.alloc,
            )
        };
    }
}

impl<T, S: Shape, A: Allocator> Owned for DynBuffer<T, S, A> {
    type Alloc = A;
    type WithConst<const N: usize> = DynBuffer<T, S::Prepend<Const<N>>, A>;

    #[inline]
    fn allocator(&self) -> &A {
        &self.alloc
    }

    #[inline]
    unsafe fn cast<U>(self) -> S::Buffer<U, A> {
        assert!(Layout::new::<T>() == Layout::new::<U>(), "layout mismatch");

        let (vec, shape) = self.into_parts();
        let (ptr, len, capacity, alloc) = vec.into_raw_parts_with_allocator();

        let vec = unsafe { Vec::from_raw_parts_in(ptr.cast(), len, capacity, alloc) };

        unsafe { Owned::from_dyn(DynBuffer::from_parts(vec, shape)) }
    }

    #[inline]
    fn clone(&self) -> Self
    where
        T: Clone,
        A: Clone,
    {
        self.as_slice().expr().cloned().eval_in(self.alloc.clone()).into_inner().into_dyn()
    }

    #[inline]
    fn clone_from(&mut self, source: &Self)
    where
        T: Clone,
        A: Clone,
    {
        self.clone_from_slice(source.as_slice());
    }

    #[inline]
    fn clone_from_slice(&mut self, slice: &Slice<T, S>)
    where
        T: Clone,
    {
        unsafe {
            self.with_mut_parts(|vec, shape| {
                if mem::needs_drop::<T>() {
                    vec.truncate(slice.len());

                    let (init, tail) = slice[..].split_at(vec.len());

                    vec.clone_from_slice(init);
                    vec.extend_from_slice(tail);
                } else {
                    vec.clear();
                    vec.extend_from_slice(&slice[..]);
                }

                shape.clone_from(slice.shape());
            });
        }
    }

    #[inline]
    fn from_dyn(buffer: DynBuffer<T, S, A>) -> Self {
        buffer
    }

    #[inline]
    fn from_static<R: ConstShape>(buffer: StaticBuffer<T, R, A>, new_shape: S) -> Self {
        assert!(new_shape.checked_len() == Some(R::default().len()), "length must not change");

        let (inner, alloc) = buffer.into_parts();
        let mut vec = <Vec<T, A>>::with_capacity_in(R::default().len(), alloc);

        unsafe {
            ptr::write(vec.as_mut_ptr().cast(), inner);
            vec.set_len(R::default().len());

            Self::from_parts(vec, new_shape)
        }
    }

    #[inline]
    fn into_buffer<B: Owned<Item = T, Alloc = A>>(self) -> B {
        let (vec, shape) = self.into_parts();

        unsafe { B::from_dyn(DynBuffer::from_parts(vec, shape.with_dims(B::Shape::from_dims))) }
    }

    #[inline]
    fn into_dyn(self) -> DynBuffer<Self::Item, Self::Shape, Self::Alloc> {
        self
    }

    #[inline]
    fn into_shape<R: Shape>(self, new_shape: R) -> R::Buffer<T, A> {
        let (vec, shape) = self.into_parts();

        unsafe { Owned::from_dyn(DynBuffer::from_parts(vec, shape.reshape(new_shape))) }
    }

    #[inline]
    fn uninit_in(shape: S, alloc: A) -> S::Buffer<MaybeUninit<T>, A> {
        let len = shape.checked_len().expect("invalid length");
        let mut vec = Vec::with_capacity_in(len, alloc);

        unsafe {
            vec.set_len(len);

            Owned::from_dyn(DynBuffer::from_parts(vec, shape))
        }
    }

    #[inline]
    fn zip_with<U, E: Expression, F>(self, expr: E, mut f: F) -> S::Buffer<U, A>
    where
        F: FnMut((T, E::Item)) -> U,
    {
        if alloc::alloc::Layout::new::<T>() == alloc::alloc::Layout::new::<U>() {
            struct DropGuard<'a, T, U, S: Shape, A: Allocator> {
                buffer: &'a mut DynBuffer<MaybeUninit<U>, S, A>,
                index: usize,
                phantom: PhantomData<T>,
            }

            impl<T, U, S: Shape, A: Allocator> Drop for DropGuard<'_, T, U, S, A> {
                #[inline]
                fn drop(&mut self) {
                    let src = self.buffer.slice.as_mut_ptr() as *mut T;
                    let dst = self.buffer.slice.as_mut_ptr() as *mut U;

                    let tail = self.buffer.slice.mapping().len() - self.index;

                    // Drop all elements except the current one, which is read but not written back.
                    unsafe {
                        if self.index > 1 {
                            ptr::slice_from_raw_parts_mut(dst, self.index - 1).drop_in_place();
                        }

                        ptr::slice_from_raw_parts_mut(src.add(self.index), tail).drop_in_place();
                    }
                }
            }

            let mut buffer = unsafe { self.cast().into_dyn() };
            let mut guard = DropGuard { buffer: &mut buffer, index: 0, phantom: PhantomData::<T> };

            let expr = guard.buffer.as_mut_slice().expr_mut().zip(expr);

            expr.for_each(|(x, y)| {
                guard.index += 1;
                _ = unsafe { x.write(f((mem::transmute_copy(x), y))) };
            });

            mem::forget(guard);

            unsafe { buffer.cast() }
        } else {
            // Need to use references to the allocator to have two active arrays.

            let (vec, shape) = self.into_parts();
            let (ptr, len, capacity, alloc) = vec.into_raw_parts_with_allocator();

            let vec = unsafe { Vec::from_raw_parts_in(ptr, len, capacity, &alloc) };
            let array = unsafe { IntoExpr::new(DynBuffer::from_parts(vec, shape).cast()) };

            let result = <Array<U, S, &A>>::with_expr_in(array.zip(expr).map(f), &alloc);

            let (vec, shape) = result.into_inner().into_dyn().into_parts();
            let (ptr, len, capacity, _) = vec.into_raw_parts_with_allocator();

            let vec = unsafe { Vec::from_raw_parts_in(ptr, len, capacity, alloc) };

            unsafe { Owned::from_dyn(DynBuffer::from_parts(vec, shape)) }
        }
    }
}

unsafe impl<T: Send, S: Shape, A: Allocator + Send> Send for DynBuffer<T, S, A> {}
unsafe impl<T: Sync, S: Shape, A: Allocator + Sync> Sync for DynBuffer<T, S, A> {}

impl<'a, T, A: Allocator> DropGuard<'a, T, A> {
    #[inline]
    fn new(vec: &'a mut Vec<T, &A>) -> Self {
        let len = vec.len();

        unsafe {
            vec.set_len(0);
        }

        Self { ptr: vec.as_mut_ptr(), len, phantom: PhantomData }
    }
}

impl<T, A: Allocator> Drop for DropGuard<'_, T, A> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            ptr::slice_from_raw_parts_mut(self.ptr, self.len).drop_in_place();
        }
    }
}

#[inline]
unsafe fn copy_dim<T, S: Shape, A: Allocator>(
    old_vec: &mut DropGuard<T, A>,
    new_vec: &mut Vec<T, &A>,
    old_dims: &[usize],
    new_dims: &[usize],
    f: &mut impl FnMut() -> T,
) {
    let old_stride: usize = old_dims[1..].iter().product();
    let new_stride: usize = new_dims[1..].iter().product();

    let old_size = old_dims[0];
    let new_size = new_dims[0];

    let min_size = old_size.min(new_size);

    unsafe {
        if old_dims.len() > 1 {
            //
            // Avoid very long compile times for release build with MIR inlining,
            // by avoiding recursion until types are known.
            //
            // This is a workaround until const if is available, see Rust issue #122301.
            //

            unsafe fn dummy<T, A: Allocator>(
                _: &mut DropGuard<T, A>,
                _: &mut Vec<T, &A>,
                _: &[usize],
                _: &[usize],
                _: &mut impl FnMut() -> T,
            ) {
                unreachable!();
            }

            let g = const {
                match S::RANK {
                    Some(..2) => dummy::<T, A>,
                    _ => copy_dim::<T, S::Tail, A>,
                }
            };

            for _ in 0..min_size {
                g(old_vec, new_vec, &old_dims[1..], &new_dims[1..], f);
            }
        } else {
            debug_assert!(old_vec.len >= min_size, "slice exceeds remainder");
            debug_assert!(new_vec.len() + min_size <= new_vec.capacity(), "slice exceeds capacity");

            ptr::copy_nonoverlapping(
                old_vec.ptr,
                new_vec.as_mut_ptr().add(new_vec.len()),
                min_size,
            );

            old_vec.ptr = old_vec.ptr.add(min_size);
            old_vec.len -= min_size;

            new_vec.set_len(new_vec.len() + min_size);
        }

        if old_size > min_size {
            let count = (old_size - min_size) * old_stride;
            let slice = ptr::slice_from_raw_parts_mut(old_vec.ptr, count);

            debug_assert!(old_vec.len >= count, "slice exceeds remainder");

            old_vec.ptr = old_vec.ptr.add(count);
            old_vec.len -= count;

            ptr::drop_in_place(slice);
        }

        let additional = (new_size - min_size) * new_stride;

        debug_assert!(new_vec.len() + additional <= new_vec.capacity(), "slice exceeds capacity");

        for _ in 0..additional {
            new_vec.as_mut_ptr().add(new_vec.len()).write(f());
            new_vec.set_len(new_vec.len() + 1);
        }
    }
}
