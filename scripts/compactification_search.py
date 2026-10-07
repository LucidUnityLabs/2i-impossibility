"""Topology calculations plus a quarantine of unsupported catalog assertions.

A necessary divisibility condition does not construct a free group action.
Euler-characteristic generation formulas are conditional on the model.
"""
from dataclasses import dataclass, asdict
from exact_2i import require
from verification_io import cli


@dataclass(frozen=True)
class Topology:
    label: str
    real_dimension: int
    euler: int
    reason: str

    def __post_init__(self):
        require(type(self.real_dimension) is int and self.real_dimension >= 0,'bad real dimension')
        require(type(self.euler) is int,'Euler characteristic must be an integer')


def product_topology(a: Topology,b: Topology) -> Topology:
    return Topology(a.label+' x '+b.label,a.real_dimension+b.real_dimension,
                    a.euler*b.euler,'Euler characteristic multiplies and real dimensions add.')


def chi_from_f_vector(f):
    require(bool(f) and all(type(n) is int and n >= 0 for n in f),'invalid cell counts')
    return sum((-1)**i*n for i,n in enumerate(f))


def free_quotient_euler(chi,order,*,free_action_verified):
    require(type(chi) is int and type(order) is int and order > 0,'bad quotient inputs')
    require(free_action_verified is True,'a free action must be supplied, not inferred from divisibility')
    require(chi%order == 0,'free-action Euler divisibility obstruction')
    return chi//order


def standard_embedding_net_generations(chi):
    require(type(chi) is int and chi%2 == 0,'CY3 Euler characteristic must be even')
    return abs(chi)//2


def build():
    """Euler arithmetic and dimensions for explicit topologies; catalog existence/action claims are not a classification."""
    cell_euler = chi_from_f_vector([120,720,1200,600])
    require(cell_euler == 0,'600-cell boundary Euler check')
    sphere = Topology('S3',3,cell_euler,'600-cell BOUNDARY triangulates S3; the filled 4-ball has Euler characteristic one.')
    quotient = Topology('S3/2I',3,free_quotient_euler(sphere.euler,120,free_action_verified=True),
                        'Left multiplication by unit quaternions is free on S3. The deck action is on the COVER.')
    factors = [Topology('T2',2,0,'product of two circles'),
               Topology('S1 x S1',2,0,'product of two circles'),
               Topology('S2',2,2,'even sphere'),Topology('S4',4,2,'even sphere')]
    products = [product_topology(quotient,x) for x in factors]
    require([x.real_dimension for x in products] == [5,5,5,7],'product dimension regression')
    require(all(x.euler == 0 for x in products),'product Euler regression')
    # Keep unsupported legacy claims visible, but never turn them into negative results.
    quarantined = [
        ('K3 with faithful 2I symplectic action','Supply the actual action/classification entry; do not confuse S5 of order 120 with SL2(5).'),
        ('K3/2I resolution','Specify a faithful action, fixed loci and resolution; free-cover arithmetic does not apply to singular quotients.'),
        ('Tian-Yau / free Z2 exclusion','Identify covering manifold, quotient, Hodge data and action. A catalog omission is not a nonexistence proof.'),
        ('CICY-7884','Pin the dataset, its indexing convention, configuration matrix and Hodge numbers.'),
        ('Quintic quotients','Specify free action and smoothness before applying chi/order.'),
        ('CY3 with 2I action','Unknown here; absence from a selected catalog is not an exhaustive theorem.'),
        ('H4 lattice/CY obstruction','Noncrystallographic four-dimensional action does not exclude higher-dimensional rational representations.'),
        ('600-cell / I','Distinguish A5 rotations from the free binary-icosahedral S3 action.')]
    conditional = free_quotient_euler(-12,2,free_action_verified=True)
    return {'topologies':[asdict(sphere),asdict(quotient)]+[asdict(x) for x in products],
            'quotient_action_on_base':'not inferred from the covering deck group',
            'S4_free_2I_action':{'excluded':2%120 != 0,'reason':'Euler divisibility, not an assertion about all finite groups'},
            'CP2_action':{'example':'diag(1,rho_2(g)) acts projectively and fixes [1:0:0]',
                          'free':False,'faithful':True},
            'conditional_CY3_Z2_example':{'assumed_parent_euler':-12,'assumed_free_action':True,
                  'quotient_euler':conditional,'standard_embedding_net_generations':standard_embedding_net_generations(conditional),
                  'scope':'Arithmetic consistency only; this does not construct a CHL model or verify an action.'},
            'quarantined':[{'claim':name,'status':'not_verified','required_evidence':why} for name,why in quarantined],
            'summary':{'quarantined_count':len(quarantined),'exhaustive':False,
                       'all_compactifications_excluded':None}}


if __name__ == '__main__':
    raise SystemExit(cli(build,__file__))
