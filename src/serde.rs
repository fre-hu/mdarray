#[cfg(feature = "nightly")]
use std::alloc::Allocator;
use std::fmt::{self, Formatter};
use std::marker::PhantomData;

use serde::de::{Error, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[cfg(not(feature = "nightly"))]
use crate::alloc::Allocator;
use crate::array::Array;
use crate::buffer::Buffer;
use crate::dim::Dim;
use crate::expr::{Expr, ExprMut, IntoExpr};
use crate::grid::Grid;
use crate::index::{Axis, Nth};
use crate::layout::Layout;
use crate::shape::{ConstShape, Shape};
use crate::span::Span;

struct GridVisitor<T, S: Shape> {
    phantom: PhantomData<(T, S)>,
}

impl<'a, T: Deserialize<'a>, S: Shape> Visitor<'a> for GridVisitor<T, S> {
    type Value = Grid<T, S>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(formatter, "an array of rank {}", S::RANK)
    }

    fn visit_seq<A: SeqAccess<'a>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        assert!(S::RANK > 0, "invalid rank");

        let mut vec = Vec::new();
        let mut dims = S::default().dims();
        let mut size = 0;

        let size_hint = seq.size_hint().unwrap_or(0);

        if S::RANK == 1 {
            vec.reserve(size_hint);

            while let Some(value) = seq.next_element()? {
                vec.push(value);
                size += 1;
            }
        } else {
            while let Some(value) = seq.next_element::<Grid<T, <Nth<0> as Axis>::Other<S>>>()? {
                if size == 0 {
                    vec.reserve(value.len() * size_hint);
                    dims[1..].copy_from_slice(&value.dims()[..]);
                } else {
                    let found = &value.dims()[..];
                    let expect = &dims[1..];

                    if found != expect {
                        let msg = format!("invalid dimensions {:?}, expected {:?}", found, expect);

                        return Err(A::Error::custom(msg));
                    }
                }

                vec.append(&mut value.into_vec());
                size += 1;
            }
        }

        if <Nth<0> as Axis>::Dim::<S>::SIZE.is_none() {
            dims[0] = size;
        } else if size != dims[0] {
            let msg = format!("invalid dimension {:?}, expected {:?}", size, dims[0]);

            return Err(A::Error::custom(msg));
        }

        Ok(Grid::from(vec).into_shape(S::from_dims(dims)))
    }
}

impl<'a, T: Deserialize<'a>, S: ConstShape> Deserialize<'a> for Array<T, S> {
    fn deserialize<R: Deserializer<'a>>(deserializer: R) -> Result<Self, R::Error> {
        if S::RANK > 0 {
            Ok(<Grid<T, S> as Deserialize>::deserialize(deserializer)?.into())
        } else {
            let value = <T as Deserialize>::deserialize(deserializer)?;

            Ok(Array::from([value]).into_shape(S::default()))
        }
    }
}

impl<'a, T: Deserialize<'a>, S: Shape> Deserialize<'a> for Grid<T, S> {
    fn deserialize<R: Deserializer<'a>>(deserializer: R) -> Result<Self, R::Error> {
        if S::RANK > 0 {
            let visitor = GridVisitor { phantom: PhantomData };

            deserializer.deserialize_seq(visitor)
        } else {
            let value = <T as Deserialize>::deserialize(deserializer)?;

            Ok(Grid::from([value]).into_shape(S::default()))
        }
    }
}

impl<T: Serialize, S: ConstShape> Serialize for Array<T, S> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}

impl<T: Serialize, S: Shape, L: Layout> Serialize for Expr<'_, T, S, L> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}

impl<T: Serialize, S: Shape, L: Layout> Serialize for ExprMut<'_, T, S, L> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}

impl<T: Serialize, S: Shape, A: Allocator> Serialize for Grid<T, S, A> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}

impl<B: Buffer<Item: Serialize>> Serialize for IntoExpr<B> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        (**self).serialize(serializer)
    }
}

impl<T: Serialize, S: Shape, L: Layout> Serialize for Span<T, S, L> {
    fn serialize<R: Serializer>(&self, serializer: R) -> Result<R::Ok, R::Error> {
        if S::RANK == 0 {
            self[S::Dims::default()].serialize(serializer)
        } else {
            let mut seq = serializer.serialize_seq(Some(self.dim(0)))?;

            for x in self.outer_expr() {
                seq.serialize_element(&x)?;
            }

            seq.end()
        }
    }
}
