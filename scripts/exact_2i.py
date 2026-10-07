"""Exact binary-icosahedral verification; no floating point or third-party imports.

Zphi(a,b) denotes a+b*phi, phi**2=phi+1. Quaternion coordinates
are Zphi pairs divided by 2. Labels are mathematical, not eigensolver indices.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
from math import lcm


class VerificationError(ValueError):
    """A claimed invariant failed its exact check."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


@dataclass(frozen=True, order=True)
class Zphi:
    a: int = 0
    b: int = 0

    def __post_init__(self):
        require(type(self.a) is int and type(self.b) is int,
                "Zphi coefficients must be Python integers")

    def __add__(self, other):
        other = as_zphi(other)
        return Zphi(self.a + other.a, self.b + other.b)

    __radd__ = __add__

    def __neg__(self):
        return Zphi(-self.a, -self.b)

    def __sub__(self, other):
        return self + -as_zphi(other)

    def __rsub__(self, other):
        return as_zphi(other) + -self

    def __mul__(self, other):
        other = as_zphi(other)
        return Zphi(self.a*other.a + self.b*other.b,
                    self.a*other.b + self.b*other.a + self.b*other.b)

    __rmul__ = __mul__

    def div(self, n: int):
        require(type(n) is int and n > 0, 'divisor must be a positive integer')
        require(self.a % n == 0 and self.b % n == 0,
                f'nonintegral Z[phi] quotient: {self} / {n}')
        return Zphi(self.a // n, self.b // n)

    def conjugate_field(self):
        """Galois conjugation phi -> 1-phi (NOT complex conjugation)."""
        return Zphi(self.a + self.b, -self.b)

    def integer(self) -> int:
        require(self.b == 0, f'not a rational integer: {self}')
        return self.a

    def json(self):
        return [self.a, self.b]


def as_zphi(x):
    if isinstance(x, Zphi):
        return x
    if type(x) is int:
        return Zphi(x)
    raise TypeError(f'expected integer or Zphi, got {type(x).__name__}')


ZERO, ONE = Zphi(), Zphi(1)
Quaternion = tuple[Zphi, Zphi, Zphi, Zphi]
Character = tuple[Zphi, ...]


def qmul(a: Quaternion, b: Quaternion) -> Quaternion:
    w,x,y,z = a
    v,r,s,t = b
    return tuple(c.div(2) for c in
                 (w*v-x*r-y*s-z*t, w*r+x*v+y*t-z*s,
                  w*s-x*t+y*v+z*r, w*t+x*s-y*r+z*v))


def qinv(a: Quaternion) -> Quaternion:
    return (a[0], -a[1], -a[2], -a[3])


def even_permutations(n: int):
    return [p for p in permutations(range(n))
            if sum(p[i] > p[j] for i in range(n) for j in range(i+1,n)) % 2 == 0]


def elements_2i() -> tuple[Quaternion, ...]:
    values = set()
    for axis, sign in product(range(4), (-1,1)):
        q = [ZERO]*4
        q[axis] = Zphi(2*sign)
        values.add(tuple(q))
    for signs in product((-1,1), repeat=4):
        values.add(tuple(Zphi(s) for s in signs))
    base = (ZERO, ONE, Zphi(0,1), Zphi(-1,1))
    for p in even_permutations(4):
        for signs in product((-1,1), repeat=4):
            values.add(tuple(signs[i]*base[p[i]] for i in range(4)))
    return tuple(sorted(values))


class BinaryIcosahedral:
    def __init__(self):
        self.elements = elements_2i()
        self.n = len(self.elements)
        require(self.n == 120, 'group order is not 120')
        lookup = {q:i for i,q in enumerate(self.elements)}
        self.identity = lookup[(Zphi(2), ZERO, ZERO, ZERO)]
        self.z = lookup[(Zphi(-2), ZERO, ZERO, ZERO)]
        require(all(sum((c*c for c in q), ZERO) == Zphi(4)
                    for q in self.elements), 'non-unit quaternion')
        self.table = []
        for a in self.elements:
            row = []
            for b in self.elements:
                p = qmul(a,b)
                require(p in lookup, 'quaternion closure failure')
                row.append(lookup[p])
            self.table.append(tuple(row))
        self.table = tuple(self.table)
        self.inverses = tuple(lookup[qinv(q)] for q in self.elements)
        e, T = self.identity, self.table
        require(all(T[e][i] == i == T[i][e] and
                    T[i][self.inverses[i]] == e == T[self.inverses[i]][i]
                    for i in range(self.n)), 'identity or inverse failure')
        require(all(len(set(row)) == self.n for row in T), 'not a Latin table')
        self.orders = tuple(self.element_order(i) for i in range(self.n))
        self.exponent = lcm(*self.orders)
        seen, classes = set(), []
        for i in range(self.n):
            if i in seen:
                continue
            cls = tuple(sorted({T[T[g][i]][self.inverses[g]] for g in range(self.n)}))
            require(not seen.intersection(cls), 'conjugacy classes overlap')
            seen.update(cls)
            classes.append(cls)
        self.classes = tuple(sorted(classes, key=lambda c:(self.orders[c[0]],len(c),c[0])))
        require(len(seen) == self.n, 'incomplete conjugacy partition')
        require(sorted(map(len,self.classes)) == [1,1,12,12,12,12,20,20,30],
                'wrong conjugacy classes')
        self.class_of = {x:c for c,cls in enumerate(self.classes) for x in cls}
        self.sizes = tuple(map(len,self.classes))
        self.id_class = self.class_of[e]
        self.z_class = self.class_of[self.z]
        require(T[self.z][self.z] == e and all(T[self.z][i] == T[i][self.z]
                    for i in range(self.n)), 'bad central involution')
        # chi_n = character of Sym^n(C^2), where chi_1(q)=2 Re(q).
        self.x = tuple(self.elements[c[0]][0] for c in self.classes)
        natural = [self.su2(n) for n in range(6)]
        self.chars = {'1':natural[0], '2a':natural[1],
                      '2b':tuple(x.conjugate_field() for x in natural[1]),
                      '3a':natural[2],
                      '3b':tuple(x.conjugate_field() for x in natural[2]),
                      '4H':natural[3], '5':natural[4], '6':natural[5]}
        self.chars['4R'] = self.mul(self.chars['2a'], self.chars['2b'])
        self.chars = {k:self.chars[k] for k in ('1','2a','2b','3a','3b','4R','4H','5','6')}
        self.dims = {k:v[self.id_class].integer() for k,v in self.chars.items()}
        require(sorted(self.dims.values()) == [1,2,2,3,3,4,4,5,6], 'wrong dimensions')
        require(sum(d*d for d in self.dims.values()) == self.n, 'incomplete character table')
        for a,ca in self.chars.items():
            for b,cb in self.chars.items():
                require(self.inner(ca,cb) == int(a == b), f'orthogonality failed: {a},{b}')
        sq = self.power_classes(2)
        self.fs = {k:self.average(tuple(v[c] for c in sq)) for k,v in self.chars.items()}
        self.kernels = {k:tuple(i for i in range(self.n)
                               if v[self.class_of[i]] == Zphi(self.dims[k]))
                        for k,v in self.chars.items()}
        self.faithful = {k:ker == (e,) for k,ker in self.kernels.items()}
        require(self.fs == {'1':1,'2a':-1,'2b':-1,'3a':1,'3b':1,'4R':1,'4H':-1,'5':1,'6':-1},
                'unexpected FS indicators')
        require(all(self.faithful[k] == (v == -1) for k,v in self.fs.items()),
                'faithfulness/quaternionicity mismatch')
        require(sum(self.fs[k]*d for k,d in self.dims.items()) ==
                sum(T[i][i] == e for i in range(self.n)) == 2, 'Frobenius count failed')

    def element_order(self, i):
        x = self.identity
        for n in range(1,self.n+1):
            x = self.table[x][i]
            if x == self.identity:
                return n
        raise VerificationError('element order exceeds group size')

    def power(self, i, n):
        require(type(n) is int and n >= 0, 'power must be a nonnegative integer')
        x = self.identity
        while n:
            if n & 1:
                x = self.table[x][i]
            i = self.table[i][i]
            n //= 2
        return x

    def power_classes(self, n):
        out = []
        for cls in self.classes:
            targets = {self.class_of[self.power(i,n)] for i in cls}
            require(len(targets) == 1, 'power map not constant on a class')
            out.append(targets.pop())
        return tuple(out)

    def su2(self, highest_weight: int) -> Character:
        require(type(highest_weight) is int and highest_weight >= 0,
                'highest weight must be a nonnegative integer (twice the spin)')
        a = tuple(ONE for _ in self.classes)
        if highest_weight == 0:
            return a
        b = self.x
        for _ in range(1,highest_weight):
            a,b = b,tuple(x*v-u for x,u,v in zip(self.x,a,b))
        return b

    def average(self, char: Character) -> int:
        require(len(char) == len(self.classes), 'character has wrong length')
        return sum((s*v for s,v in zip(self.sizes,char)), ZERO).div(self.n).integer()

    def inner(self, a, b):
        # All 2I character values are real; field conjugation is NOT used here.
        return self.average(self.mul(a,b))

    @staticmethod
    def mul(a,b):
        require(len(a) == len(b), 'character length mismatch')
        return tuple(x*y for x,y in zip(a,b))

    @staticmethod
    def add(*chars):
        require(bool(chars) and len({len(c) for c in chars}) == 1, 'invalid character sum')
        return tuple(sum(x,ZERO) for x in zip(*chars))

    @staticmethod
    def scale(n, a):
        require(type(n) is int, 'integer character coefficient required')
        return tuple(n*x for x in a)

    def decompose(self, char):
        mult = {k:self.inner(char,v) for k,v in self.chars.items()}
        require(all(m >= 0 for m in mult.values()), 'negative character multiplicity')
        reconstructed = self.add(*(self.scale(mult[k],v) for k,v in self.chars.items()))
        require(reconstructed == char, 'character reconstruction failed')
        require(sum(mult[k]*d for k,d in self.dims.items()) == char[self.id_class].integer(),
                'decomposition dimension mismatch')
        return {k:m for k,m in mult.items() if m}

    def invariant_spin(self, twice_spin):
        result = self.average(self.su2(twice_spin))
        require(result >= 0, 'negative invariant multiplicity')
        return result

    def certificate(self):
        return {'order':self.n, 'exponent':self.exponent,
                'order_counts':{str(k):v for k,v in sorted(Counter(self.orders).items())},
                'classes':[{'order':self.orders[c[0]],'size':len(c),
                            'representative':[x.json() for x in self.elements[c[0]]]}
                           for c in self.classes],
                'irreps':{k:{'dimension':self.dims[k], 'fs':self.fs[k],
                             'faithful':self.faithful[k],
                             'values':[x.json() for x in v]}
                           for k,v in self.chars.items()},
                'coordinate_convention':'quaternion entries are (a+b*phi)/2; character entries a+b*phi'}


@lru_cache(maxsize=1)
def group() -> BinaryIcosahedral:
    return BinaryIcosahedral()
