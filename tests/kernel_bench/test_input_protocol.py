"""Standalone launchers seed and record the same invocation-owned streams."""
import copy
import pytest
from kernel_bench.input_protocol import InputProtocol
from tests.kernel_bench.test_input_seeds import _cell

class Generator:
    def manual_seed(self, seed):
        self.seed = seed

def test_stream_seeds_are_reproducible_and_owned_by_the_cell():
    a,b=InputProtocol(_cell()),InputProtocol(_cell())
    g,h=Generator(),Generator()
    for name,rank,slot in [('weights',None,0),('weights',0,0),('weights',1,0),
                            ('input',None,0),('input',None,1),('check',None,0)]:
        a.manual_seed(g,name,rank=rank,slot=slot)
        b.manual_seed(h,name,rank=rank,slot=slot)
        assert g.seed==h.seed
        assert str(g.seed)==a.streams[-1]['seed']
    assert len({e['seed'] for e in a.streams})==6
    assert a.evidence()==b.evidence()
    InputProtocol.validate(a.invocation,a.evidence())
    other=InputProtocol(_cell(dims={'B':2}))
    other.manual_seed(g,'weights')
    assert other.streams[0]['seed']!=a.streams[0]['seed']

def test_export_refuses_an_unobserved_or_changed_seed_protocol():
    protocol=InputProtocol(_cell())
    with pytest.raises(ValueError,match='no generator'):
        protocol.evidence()
    protocol.manual_seed(Generator(),'input')
    for field,value in [('input_seed','0'),('physical_key',[]),('streams',[])]:
        evidence={**protocol.evidence(),field:value}
        with pytest.raises(ValueError):InputProtocol.validate(protocol.invocation,evidence)
    changed=copy.deepcopy(protocol.evidence());changed['streams'][0]['seed']='0'
    with pytest.raises(ValueError,match='stream seed'):
        InputProtocol.validate(protocol.invocation,changed)

def test_row_writer_checks_standalone_protocol_before_stamping_its_seed():
    from kernel_bench.emit import build_row
    with pytest.raises(ValueError,match='recorded input protocol'):
        build_row(_cell(),None,None,None,date='2026-11-16',contributor='test',
                  input_protocol={'input_seed':'0'})
