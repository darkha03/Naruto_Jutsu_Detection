jutsu_chains = {
    "Fireball": ["Monkey", "Tiger", "Dragon"],
    "Water Dragon": ["Snake", "Horse", "Dragon"],
}


class Candidate:
    def __init__(self, chain_name):
        self.chain_name = chain_name
        self.index = 1

    def expected_next(self):
        sequence = jutsu_chains[self.chain_name]
        if self.index >= len(sequence):
            return None
        return sequence[self.index]

    def matched_sequence(self):
        return jutsu_chains[self.chain_name][: self.index]

    def advance(self):
        self.index += 1


class Chainer:
    def __init__(self):
        self.last_class_name = None
        self.candidates = []
        self.last_active_sequence = []

    def update(self, class_name):
        if class_name is None:
            completed_sequence = list(self.last_active_sequence) if self.last_active_sequence else None
            self.clear()
            return {
                "active_sequence": [],
                "completed_sequence": completed_sequence,
                "reset": True,
                "is_chain_completed": False,
                "completed_chain_name": None,
            }

        if class_name == self.last_class_name:
            completion_info = self.validate_completed_chain()
            return {
                "active_sequence": list(self.last_active_sequence),
                "completed_sequence": None,
                "reset": False,
                "is_chain_completed": completion_info["is_completed"],
                "completed_chain_name": completion_info["chain_name"],
            }

        self.last_class_name = class_name
        self.update_candidates(class_name)
        if not self.candidates:
            self._start_new_candidates(class_name)

        best_candidate = self._best_candidate()
        self.last_active_sequence = best_candidate.matched_sequence() if best_candidate is not None else []
        completion_info = self.validate_completed_chain()

        return {
            "active_sequence": list(self.last_active_sequence),
            "completed_sequence": None,
            "reset": False,
            "is_chain_completed": completion_info["is_completed"],
            "completed_chain_name": completion_info["chain_name"],
        }

    def get_active_sequence(self):
        return list(self.last_active_sequence)

    def clear(self):
        self.candidates = []
        self.last_active_sequence = []
        self.last_class_name = None

    def _start_new_candidates(self, class_name):
        for chain_name, sequence in jutsu_chains.items():
            if sequence and sequence[0] == class_name:
                self.candidates.append(Candidate(chain_name))

    def update_candidates(self, class_name):
        surviving_candidates = []
        for candidate in self.candidates:
            expected = candidate.expected_next()
            if expected == class_name:
                candidate.advance()
                surviving_candidates.append(candidate)
        self.candidates = surviving_candidates

    def _best_candidate(self):
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda candidate: candidate.index)

    def validate_completed_chain(self):
        best_candidate = self._best_candidate()
        if best_candidate is None:
            return {"is_completed": False, "chain_name": None}

        chain_sequence = jutsu_chains[best_candidate.chain_name]
        is_completed = best_candidate.index >= len(chain_sequence)
        return {
            "is_completed": is_completed,
            "chain_name": best_candidate.chain_name if is_completed else None,
        }

