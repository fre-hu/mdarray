#[cfg(feature = "nightly")]
use alloc::alloc::Allocator;
#[cfg(not(feature = "std"))]
use alloc::format;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::fmt::{self, Formatter};
use core::marker::PhantomData;

use serde::de::{Error, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(not(feature = "nightly"))]
use crate::allocator::Allocator;
use crate::array::Array;
use crate::dim::Dim;
use crate::layout::Layout;
use crate::shape::{ConstShape, Shape};
use crate::slice::Slice;
use crate::tensor::Tensor;
use crate::view::{View, ViewMut};
use crate::{array, tensor};

struct TensorVisitor<T, S: Shape> {
    phantom: PhantomData<(T, S)>,
}

impl<'a, T: Deserialize<'a>, S: Shape> Visitor<'a> for TensorVisitor<T, S> {
    type Value = Tensor<T, S>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(formatter, "an array of rank {}", S::RANK.expect("invalid rank"))
    }

    fn visit_seq<A: SeqAccess<'a>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        assert!(S::RANK.is_some_and(|rank| rank > 0), "invalid rank");

        let mut vec = Vec::new();
        let mut shape = S::default();
        let mut size = 0;

        let size_hint = seq.size_hint().unwrap_or(0);

        if S::RANK == Some(1) {
            vec.reserve(size_hint);

            while let Some(value) = seq.next_element()? {
                vec.push(value);
                size += 1;
            }
        } else {
            while let Some(value) = seq.next_element::<Tensor<T, S::Tail>>()? {
                if size == 0 {
                    vec.reserve(value.len() * size_hint);
                    shape.with_mut_dims(|dims| {
                        value.shape().with_dims(|src| dims[1..].copy_from_slice(src));
                    });
                } else {
                    shape.with_dims(|dims| {
                        value.shape().with_dims(|src| {
                            let dst = &dims[1..];

                            if src != dst {
                                let msg = format!("invalid dimensions {src:?}, expected {dst:?}");

                                Err(A::Error::custom(msg))
                            } else {
                                Ok(())
                            }
                        })
                    })?;
                }

                vec.append(&mut value.into_vec());
                size += 1;
            }
        }

        if S::Head::SIZE.is_none() {
            shape.with_mut_dims(|dims| dims[0] = size);
        } else if size != shape.dim(0) {
            let msg = format!("invalid dimension {size:?}, expected {:?}", shape.dim(0));

            return Err(A::Error::custom(msg));
        }

        Ok(Tensor::from(vec).into_shape(shape))
    }
}

impl<'a, T: Deserialize<'a>, S: ConstShape> Deserialize<'a> for Array<T, S> {
    fn deserialize<R: Deserializer<'a>>(deserializer: R) -> Result<Self, R::Error> {
        if S::RANK != Some(0) {
            Ok(<Tensor<T, S> as Deserialize>::deserialize(deserializer)?.into())
        } else {
            let value = <T as Deserialize>::deserialize(deserializer)?;

            Ok(array![value].into_shape())
        }
    }
}

impl<'a, T: Deserialize<'a>, S: Shape> Deserialize<'a> for Tensor<T, S> {
    fn deserialize<R: Deserializer<'a>>(deserializer: R) -> Result<Self, R::Error> {
        let rank = S::RANK.expect("dynamic rank not supported");

        if rank > 0 {
            let visitor = TensorVisitor { phantom: PhantomData };

            deserializer.deserialize_seq(visitor)
        } else {
            let value = <T as Deserialize>::deserialize(deserializer)?;

            Ok(tensor![value].into_shape(S::default()))
        }
    }
}

impl<T: Serialize, S: ConstShape> Serialize for Array<T, S> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}

impl<T: Serialize, S: Shape, L: Layout> Serialize for Slice<T, S, L> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        let rank = S::RANK.expect("dynamic rank not supported");

        if rank == 0 {
            self[[]].serialize(serializer)
        } else {
            let mut seq = serializer.serialize_seq(Some(self.dim(0)))?;

            for x in self.outer_expr() {
                seq.serialize_element(&x)?;
            }

            seq.end()
        }
    }
}

impl<T: Serialize, S: Shape, A: Allocator> Serialize for Tensor<T, S, A> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}

impl<T: Serialize, S: Shape, L: Layout> Serialize for View<'_, T, S, L> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}

impl<T: Serialize, S: Shape, L: Layout> Serialize for ViewMut<'_, T, S, L> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}
