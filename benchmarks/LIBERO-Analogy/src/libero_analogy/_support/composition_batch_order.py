"""Strict ordered predicate events, with re-arming for initially true Close."""
import re

def parse_atom(atom):
    match=re.fullmatch(r'\s*(\w+)\s*\((.*?)\)\s*',atom)
    if match is None:raise ValueError(atom)
    return (match[1].lower(),*(s.strip() for s in match[2].split(',') if s.strip()))

class OrderedEvents:
    def __init__(self,atoms):
        self.atoms=list(atoms);self.initial=None;self.previous=None
        self.event_steps=[None]*len(atoms);self.stage_index=0;self.violation=None
    def update(self,step,values):
        values=[bool(v) for v in values]
        if self.initial is None:
            if step!=0:raise RuntimeError('Missing initial event audit')
            self.initial=values;self.previous=values
            if values[0]:raise RuntimeError('First event already true initially')
            if any(v and parse_atom(a)[0]!='close' for a,v in zip(self.atoms,values)):
                raise RuntimeError('Non-close event already true initially')
            return
        def armed(i):
            atom=parse_atom(self.atoms[i])
            if atom[0]!='close':return True
            # A closed drawer may settle around Close's strict q=0 boundary.
            # Re-arm its terminal closure only after the requested Open
            # milestone, not after a sub-millimetre predicate fluctuation.
            preceding=[j for j,a in enumerate(self.atoms[:i]) if parse_atom(a)==('open',*atom[1:])]
            if preceding:return all(self.event_steps[j] is not None for j in preceding)
            if self.initial[i]:raise RuntimeError('Initially true Close lacks a preceding Open event')
            return True
        rising=[i for i,(before,now) in enumerate(zip(self.previous,values)) if now and not before and self.event_steps[i] is None and armed(i)]
        expected=self.stage_index
        if any(i>expected for i in rising):self.violation=self.violation or f'later event occurred before its turn: step={step}, expected={expected}, rising={rising}'
        if expected<len(values) and expected in rising and not self.violation:
            self.event_steps[expected]=int(step);self.stage_index+=1
        self.previous=values
    @property
    def complete(self):return self.stage_index==len(self.atoms) and not self.violation
    def as_dict(self):
        return dict(ordered_event_atoms=self.atoms,initial_event_values=self.initial,event_first_steps=self.event_steps,
                    completed_event_count=self.stage_index,order_violation=self.violation,ordered_completion_observed=self.complete)

def self_test():
    tracker=OrderedEvents(['open(drawer)','in(bowl,drawer)','close(drawer)'])
    for event in [(0,[False,False,True]),(10,[True,False,False]),(20,[True,True,False]),(30,[False,True,True])]:tracker.update(*event)
    assert tracker.complete and tracker.event_steps==[10,20,30]
    for events in [[(1,[False,True]),(2,[True,True])],[(1,[True,True])]]:
        bad=OrderedEvents(['in(a,x)','in(b,y)']);bad.update(0,[False,False])
        for event in events:bad.update(*event)
        assert bad.violation and not bad.complete
    early=OrderedEvents(['open(x)','in(a,x)','close(x)']);early.update(0,[False,False,True]);early.update(1,[False,False,False]);early.update(2,[False,False,True]);assert not early.violation
    early.update(3,[True,False,False]);early.update(4,[False,False,True]);assert early.violation
    jitter=OrderedEvents(['open(x)','in(a,x)','close(x)']);jitter.update(0,[False,False,False]);jitter.update(1,[False,False,True]);assert not jitter.violation

if __name__=='__main__':self_test();print('Ordered event tests passed')
