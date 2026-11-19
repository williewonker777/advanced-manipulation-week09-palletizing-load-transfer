"""Connect two observation paths only at an evidenced handoff."""


class SlowHandoff:
    def __init__(self):
        self.previous_aux = False
        self.active = False
        self.quiet = 0

    def update(self, aux, slow_evidence, incumbent_on, clear_required, armed):
        # The fast innovation that triggered the auxiliary path is not independent
        # confirmation; it cannot start a bridge or reset its clearance counter.
        if not armed:
            self.__init__()
            return False
        falling = self.previous_aux and not aux
        self.previous_aux = bool(aux)
        if incumbent_on:
            self.active = False
        elif falling and slow_evidence:
            self.active = True
        if self.active:
            self.quiet = 0 if slow_evidence else self.quiet + 1
            if self.quiet >= clear_required:
                self.active = False
        if not self.active:
            self.quiet = 0
        return self.active
