"""The shared Bloom-filter backend used by all three final pipelines."""

from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass

import numpy as np


# This is an experimental domain-separation label, not a password.  It is kept
# fixed because changing it changes every Bloom address and therefore every
# recorded score.
THESIS_KEY = b"masters-thesis-revival-hbf-v1"


@dataclass(frozen=True)
class Block:
    start: int
    end: int
    name: str
    level: str = "lowest"

    @property
    def length(self) -> int:
        return self.end - self.start


def bloom_parameters(population: int, false_positive_probability: float) -> tuple[int, int]:
    """Standard optimal Bloom size and number of hashes for n inserted values."""
    if population <= 0 or not 0.0 < false_positive_probability < 1.0:
        raise ValueError("Population must be positive and probability must lie in (0, 1)")
    m = math.ceil(-population * math.log(false_positive_probability) / math.log(2.0) ** 2)
    k = max(1, round((m / population) * math.log(2.0)))
    return m, k


def fnv1a_32(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def encode_blocks(bits: np.ndarray, layout: list[Block]) -> np.ndarray:
    """Convert each binary block to an integer without losing long P1 blocks."""
    bits = np.asarray(bits, dtype=np.bool_)
    if bits.ndim != 2:
        raise ValueError("Templates must be a two-dimensional image-by-bit array")
    codes = np.empty((len(bits), len(layout)), dtype=object)
    for column, block in enumerate(layout):
        packed = np.packbits(bits[:, block.start:block.end], axis=1, bitorder="little")
        # Build a real list here.  Assigning the generator itself would place
        # one generator object in every cell of NumPy's object column.
        codes[:, column] = [int.from_bytes(row.tobytes(), "little") for row in packed]
    return codes


def original_address(message_bits: np.ndarray, key_number: int, filter_size: int) -> int:
    """Legacy address sequence retained for original-code compatibility studies.

    The published implementation represented each block and small hash key as
    text bits, SHA-256 hashed their concatenation, then FNV-1a hashed the hex
    digest and XOR-folded it to the power-of-two filter width.  Final thesis
    scores do *not* use this function; they use the verified routine below.
    """
    if filter_size <= 1:
        raise ValueError("The legacy routine expects a filter with at least two positions")
    message = "".join("1" if value else "0" for value in message_bits)
    small_key = format(key_number, "08b")
    digest_hex = hashlib.sha256((message + small_key).encode("ascii")).hexdigest()
    raw = fnv1a_32(digest_hex.encode("ascii"))
    width = int(math.log2(filter_size))
    # The reference program used a 2^floor(log2(m))-1 mask even when m itself
    # was not a power of two (its example m was 140,530).  Consequently the
    # highest positions of such a Bloom array were never addressed.  This odd
    # detail is retained because compatibility means matching the code, not
    # silently repairing it.
    mask = (1 << width) - 1
    return (raw ^ (raw >> width)) & mask


class BloomBank:
    """One equal-sized Bloom filter per template block.

    Each filter stores values from all enrollment photographs in the dataset.
    A query score is simply the number of filters reporting membership.
    """

    def __init__(self, layout: list[Block], population: int, probability: float) -> None:
        self.layout = layout
        self.population = population
        self.probability = probability
        self.m, self.k = bloom_parameters(population, probability)
        self.bank = np.zeros((len(layout) * self.m + 7) // 8, dtype=np.uint8)
        self._address_cache: dict[tuple[int, int], tuple[int, ...]] = {}

    def _addresses(self, column: int, code: int) -> tuple[int, ...]:
        cache_key = (column, code)
        if cache_key in self._address_cache:
            return self._address_cache[cache_key]
        block = self.layout[column]
        packed = int(code).to_bytes((block.length + 7) // 8, "little")
        # Level, block name, and length are included so equal bit strings in
        # different positions do not share an address domain by accident.
        domain = (
            block.level.encode() + b"\0" + block.name.encode() + b"\0"
            + block.length.to_bytes(8, "big") + packed
        )
        addresses = tuple(
            fnv1a_32(hmac.new(THESIS_KEY, counter.to_bytes(4, "big") + domain, hashlib.sha256).digest())
            % self.m
            for counter in range(self.k)
        )
        self._address_cache[cache_key] = addresses
        return addresses

    def _global(self, column: int, address: int) -> int:
        return column * self.m + address

    def _set(self, index: int) -> None:
        self.bank[index >> 3] |= np.uint8(1 << (index & 7))

    def _get(self, index: int) -> bool:
        return bool(self.bank[index >> 3] & np.uint8(1 << (index & 7)))

    def enroll(self, codes: np.ndarray, indices: list[int]) -> None:
        if len(indices) != self.population:
            raise ValueError("Declared population does not match enrollment templates")
        for column in range(codes.shape[1]):
            # Duplicate block values do not need repeated insertion.
            for code in np.unique(codes[indices, column]):
                for address in self._addresses(column, int(code)):
                    self._set(self._global(column, address))

    def query(self, codes: np.ndarray, indices: list[int]) -> np.ndarray:
        scores = np.zeros(len(indices), dtype=np.int32)
        for column in range(codes.shape[1]):
            unique, inverse = np.unique(codes[indices, column], return_inverse=True)
            matches = np.fromiter(
                (
                    all(self._get(self._global(column, address)) for address in self._addresses(column, int(code)))
                    for code in unique
                ),
                dtype=np.bool_,
                count=len(unique),
            )
            scores += matches[inverse]
        return scores

    @property
    def storage_bytes(self) -> int:
        return len(self.bank)

    def occupancy(self) -> float:
        used = np.unpackbits(self.bank, bitorder="little")[:len(self.layout) * self.m]
        return float(used.mean())


class OriginalBloomBank:
    """Readable equivalent of the supplied 2019 HBF enrollment/query loops.

    Use this with `pipeline1.layout("original")` for the original 165-filter
    behavior.  The final thesis comparison instead uses `BloomBank` with the
    corrected layout.  Keeping separate classes makes that methodological
    distinction impossible to hide behind a flag in a result table.
    """

    def __init__(
        self, layout: list[Block], filter_size: int = 140_530, hash_count: int = 4
    ) -> None:
        if filter_size <= 1 or hash_count <= 0:
            raise ValueError("Legacy Bloom size and hash count must be positive")
        self.layout = layout
        self.filter_size = filter_size
        self.hash_count = hash_count
        self.bank = np.zeros((len(layout), filter_size), dtype=np.bool_)

    def _positions(self, block_bits: np.ndarray) -> tuple[int, ...]:
        return tuple(
            original_address(block_bits, key, self.filter_size)
            for key in range(1, self.hash_count + 1)
        )

    def enroll(self, bits: np.ndarray, indices: list[int]) -> None:
        templates = np.asarray(bits, dtype=np.bool_)
        for column, block in enumerate(self.layout):
            for index in indices:
                positions = self._positions(templates[index, block.start:block.end])
                self.bank[column, list(positions)] = True

    def query(self, bits: np.ndarray, indices: list[int]) -> np.ndarray:
        templates = np.asarray(bits, dtype=np.bool_)
        scores = np.zeros(len(indices), dtype=np.int32)
        for row, index in enumerate(indices):
            for column, block in enumerate(self.layout):
                positions = self._positions(templates[index, block.start:block.end])
                scores[row] += int(bool(np.all(self.bank[column, list(positions)])))
        return scores
