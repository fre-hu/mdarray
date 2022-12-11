use std::fmt::{self, Formatter};
use std::marker::PhantomData;

use serde::de::{Error, SeqAccess, Unexpected, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::buffer::Buffer;
use crate::dim::{Dim, Rank};
use crate::format::Format;
use crate::grid::{DenseGrid, GridBase};
use crate::span::SpanBase;

struct GridVisitor<T, D: Dim> {
    phantom: PhantomData<(T, D)>,
}

impl<'a, T: Deserialize<'a>, D: Dim> Visitor<'a> for GridVisitor<T, D>
where
    DenseGrid<T, D::Lower>: Deserialize<'a>,
{
    type Value = DenseGrid<T, D>;

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

                Ok(DenseGrid::<T, Rank<1, D::Order>>::from(vec).into_shape(shape))
            } else {
                Ok(DenseGrid::new())
            }
        } else if let Some(value) = seq.next_element::<DenseGrid<T, D::Lower>>()? {
            if value.is_empty() {
                Err(A::Error::invalid_length(value.len(), &self))
            } else {
                let capacity = value.len() * seq.size_hint().unwrap_or(0);
                let shape = value.shape();

                let mut grid = DenseGrid::with_capacity(capacity);
                let mut larger = D::Shape::default();

                larger[D::dims(..D::RANK - 1)].copy_from_slice(&shape[..]);
                larger[D::dim(D::RANK - 1)] = 1;

                grid.append(&mut value.into_shape(larger));

                while let Some(value) = seq.next_element::<DenseGrid<T, D::Lower>>()? {
                    if value.shape()[..] != shape[..] {
                        return Err(A::Error::invalid_value(Unexpected::Seq, &self));
                    }

                    grid.append(&mut value.into_shape(larger));
                }

                Ok(grid)
            }
        } else {
            Ok(DenseGrid::new())
        }
    }
}

impl<'a, T: Deserialize<'a>, D: Dim> Deserialize<'a> for DenseGrid<T, D> {
    fn deserialize<S: Deserializer<'a>>(deserializer: S) -> Result<Self, S::Error> {
        let visitor = GridVisitor { phantom: PhantomData };

        deserializer.deserialize_seq(visitor)
    }
}

impl<B: Buffer> Serialize for GridBase<B>
where
    SpanBase<B::Item, B::Dim, B::Format>: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_span().serialize(serializer)
    }
}

impl<T: Serialize, D: Dim, F: Format> Serialize for SpanBase<T, D, F> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if D::RANK == 0 {
            self[D::Shape::default()].serialize(serializer)
        } else {
            let len = if self.is_empty() { 0 } else { self.size(D::dim(D::RANK - 1)) };
            let mut seq = serializer.serialize_seq(Some(len))?;

            if len > 0 {
                if D::RANK == 1 {
                    for x in self.flatten().iter() {
                        seq.serialize_element(x)?;
                    }
                } else {
                    for x in self.outer_iter() {
                        seq.serialize_element(&x)?;
                    }
                }
            }

            seq.end()
        }
    }
}
