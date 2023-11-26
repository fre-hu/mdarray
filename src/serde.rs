use std::fmt::{self, Formatter};
use std::marker::PhantomData;

use serde::de::{Error, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::array::{Array, GridArray};
use crate::buffer::Buffer;
use crate::dim::{Const, Dim};

struct GridVisitor<T, D: Dim> {
    phantom: PhantomData<(T, D)>,
}

impl<'a, T: Deserialize<'a>, D: Dim> Visitor<'a> for GridVisitor<T, D>
where
    GridArray<T, D::Lower>: Deserialize<'a>,
{
    type Value = GridArray<T, D>;

    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
        write!(formatter, "an array of rank {}", D::RANK)
    }

    fn visit_seq<A: SeqAccess<'a>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        assert!(D::RANK > 0, "invalid rank");

        if D::RANK == 1 {
            if let Some(value) = seq.next_element()? {
                let mut vec = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                let mut shape = D::Shape::default();

                vec.push(value);

                while let Some(value) = seq.next_element()? {
                    vec.push(value);
                }

                shape[0] = vec.len();

                Ok(GridArray::<T, Const<1>>::from(vec).into_shape(shape))
            } else {
                Ok(GridArray::new())
            }
        } else if let Some(value) = seq.next_element::<GridArray<T, D::Lower>>()? {
            if value.is_empty() {
                Err(A::Error::custom("inner sequence must be non-empty"))
            } else {
                let capacity = value.len() * seq.size_hint().unwrap_or(0);
                let expected = value.shape();

                let mut grid = GridArray::<T, D>::with_capacity(capacity);
                let mut larger = D::Shape::default();

                larger[..D::RANK - 1].copy_from_slice(&expected[..]);
                larger[D::RANK - 1] = 1;

                grid.append(&mut value.into_shape(larger));

                while let Some(value) = seq.next_element::<GridArray<T, D::Lower>>()? {
                    let shape = value.shape();

                    if shape[..] != expected[..] {
                        let msg = format!("invalid shape {:?}, expected {:?}", shape, expected);

                        return Err(A::Error::custom(msg));
                    }

                    grid.append(&mut value.into_shape(larger));
                }

                Ok(grid)
            }
        } else {
            Ok(GridArray::new())
        }
    }
}

impl<'a, T: Deserialize<'a>, D: Dim> Deserialize<'a> for GridArray<T, D> {
    fn deserialize<S: Deserializer<'a>>(deserializer: S) -> Result<Self, S::Error> {
        let visitor = GridVisitor { phantom: PhantomData };

        deserializer.deserialize_seq(visitor)
    }
}

impl<T: Serialize, B: Buffer<Item = T> + ?Sized> Serialize for Array<B> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if B::Dim::RANK == 0 {
            self[<B::Dim as Dim>::Shape::default()].serialize(serializer)
        } else {
            let dim = B::Dim::RANK - 1;
            let len = if self.as_span().is_empty() { 0 } else { self.as_span().size(dim) };

            let mut seq = serializer.serialize_seq(Some(len))?;

            // Empty arrays should give an empty sequence.
            if len > 0 {
                for x in self.as_span().outer_iter() {
                    seq.serialize_element(&x)?;
                }
            }

            seq.end()
        }
    }
}
