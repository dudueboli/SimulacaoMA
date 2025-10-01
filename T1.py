import math
import heapq
import argparse
import random
import yaml  
from collections import defaultdict, deque


class RandomStream:
    def __init__(self, numbers):
        self.numbers = numbers
        self.i = 0
        self.total_used = 0

    def next_u01(self):
        if self.i >= len(self.numbers):
            raise RuntimeError("Acabaram os aleatórios antes da hora.")
        u = self.numbers[self.i]
        self.i += 1
        self.total_used += 1
        return u

    def uniform(self, a, b):
        return a + (b - a) * self.next_u01()

    def choice_by_prob(self, weighted_targets):
        """
        weighted_targets: list of (target, p). p's devem somar 1 (tolerância numérica).
        Consome 1 aleatório.
        """
        u = self.next_u01()
        acc = 0.0
        for tgt, p in weighted_targets:
            acc += p
            if u <= acc or abs(acc - 1.0) < 1e-12:
                return tgt
        return weighted_targets[-1][0]

ARRIVAL = 1
DEPARTURE = 2

_event_seq = 0
def _next_seq():
    global _event_seq
    _event_seq += 1
    return _event_seq

class Event:
    __slots__ = ("time","etype","queue","seq","payload")
    def __init__(self, time, etype, queue, payload=None):
        self.time = time
        self.etype = etype
        self.queue = queue
        self.seq = _next_seq()
        self.payload = payload  # client id etc.

    def __lt__(self, other):
        # prioridade por tempo; desempate estável por seq
        if self.time == other.time:
            return self.seq < other.seq
        return self.time < other.time

class QueueNode:
    def __init__(self, name, cfg):
        self.name = name
        self.servers = int(cfg.get("servers", 1))
        # capacidade == número máximo na ESPERA (buffer).
        cap = cfg.get("capacity", None)
        self.capacity = math.inf if cap is None else int(cap)
        self.min_service = float(cfg.get("minService"))
        self.max_service = float(cfg.get("maxService"))
        self.has_external = ("minInterarrival" in cfg) and ("maxInterarrival" in cfg)
        self.min_inter = float(cfg.get("minInterarrival", 0.0))
        self.max_inter = float(cfg.get("maxInterarrival", 0.0))
        self.first_arrival = float(cfg.get("firstArrival", 0.0)) if self.has_external else None

        # estado
        self.in_service = 0
        self.queue = deque()
        self.losses = 0

        # métricas de tempo por estado 
        self.state_time = defaultdict(float)
        self.last_time = 0.0

        self.area_num_in_system = 0.0
        self.area_num_in_queue = 0.0

        self.busy_time = 0.0

        self.completed = 0
        self.arrived = 0

    def sample_service(self, rnd: RandomStream):
        return rnd.uniform(self.min_service, self.max_service)

    def sample_interarrival(self, rnd: RandomStream):
        return rnd.uniform(self.min_inter, self.max_inter)

    def num_in_system(self):
        return self.in_service + len(self.queue)

    def update_times(self, now):
        dt = now - self.last_time
        if dt < 0: 
            dt = 0
        n = self.num_in_system()
        self.state_time[n] += dt
        self.area_num_in_system += n * dt
        self.area_num_in_queue += len(self.queue) * dt
        self.busy_time += self.in_service * dt
        self.last_time = now

class NetworkSim:
    def __init__(self, model, rnd: RandomStream, stop_after_randoms=None):
        self.queues = {}        
        self.routing = defaultdict(list)  
        self.rnd = rnd
        self.stop_after_randoms = stop_after_randoms
        self.time = 0.0
        self.event_list = []
        self.global_last_time = 0.0
        self.total_time = 0.0
        self.client_seq = 0

        self.parse_model(model)

    def parse_model(self, m):
        for qname, cfg in m["queues"].items():
            self.queues[qname] = QueueNode(qname, cfg)

        for arc in m["network"]:
            s = arc["source"] 
            t = arc["target"]
            p = float(arc["probability"])
            self.routing[s].append((t, p))

        for s, lst in self.routing.items():
            ssum = sum(p for _, p in lst)
            if abs(ssum - 1.0) > 1e-6:
                raise ValueError(f"Probabilidades de roteamento de {s} somam {ssum:.6f} (≠ 1).")

        for q in self.queues.values():
            q.last_time = 0.0
            if q.has_external:
                t0 = q.first_arrival
                heapq.heappush(self.event_list, Event(t0, ARRIVAL, q.name, payload={"external": True}))

    def next_client_id(self):
        self.client_seq += 1
        return self.client_seq

    def should_stop(self):
        return (self.stop_after_randoms is not None) and (self.rnd.total_used >= self.stop_after_randoms)

    def schedule_external_arrival(self, q: QueueNode, now):
        # agenda próxima chegada externa 
        if self.should_stop(): 
            return
        ia = q.sample_interarrival(self.rnd)
        heapq.heappush(self.event_list, Event(now + ia, ARRIVAL, q.name, payload={"external": True}))

    def schedule_departure(self, q: QueueNode, now):
        if self.should_stop():
            return
        svc = q.sample_service(self.rnd)  # 1 aleatorio
        cid = self.next_client_id()
        heapq.heappush(self.event_list, Event(now + svc, DEPARTURE, q.name, payload={"cid": cid}))

    def run(self):
        while self.event_list:
            ev = heapq.heappop(self.event_list)
            now = ev.time
            self.time = now

            for q in self.queues.values():
                q.update_times(now)
            self.global_last_time = now

            if ev.etype == ARRIVAL:
                self.handle_arrival(ev)
            else:
                self.handle_departure(ev)

            if self.should_stop():
                break

        for q in self.queues.values():
            q.update_times(self.time)
        self.total_time = self.time

    def handle_arrival(self, ev: Event):
        q = self.queues[ev.queue]
        q.arrived += 1

        if ev.payload and ev.payload.get("external"):
            if q.has_external:
                self.schedule_external_arrival(q, self.time)

        if q.in_service < q.servers:
            q.in_service += 1
            self.schedule_departure(q, self.time)  
        else:
            if len(q.queue) < q.capacity:
                q.queue.append(self.next_client_id())
            else:
                q.losses += 1  

    def handle_departure(self, ev: Event):
        q = self.queues[ev.queue]
        q.completed += 1

        if len(q.queue) > 0:
            _ = q.queue.popleft()
            self.schedule_departure(q, self.time)  
        else:
            q.in_service -= 1
            if q.in_service < 0: 
                q.in_service = 0


        if ev.queue in self.routing:
            if self.should_stop(): 
                return
            tgt = self.rnd.choice_by_prob(self.routing[ev.queue])
            if tgt != "OUT":
                heapq.heappush(self.event_list, Event(self.time, ARRIVAL, tgt, payload={"external": False}))

def load_model(path):
    if yaml is None:
        raise SystemExit("PyYAML não encontrado. Instale com: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    model = yaml.safe_load(raw)

    cfg = {
        "queues": {},
        "network": []
    }

    for qname, qcfg in model.get("queues", {}).items():
        entry = {
            "servers": qcfg.get("servers", 1),
            "capacity": qcfg.get("capacity", None),
            "minService": qcfg.get("minService"),
            "maxService": qcfg.get("maxService"),
        }
        if "minArrival" in qcfg or "maxArrival" in qcfg:
            entry["minInterarrival"] = qcfg.get("minArrival")
            entry["maxInterarrival"] = qcfg.get("maxArrival")
        if "firstArrival" in qcfg:
            entry["firstArrival"] = qcfg.get("firstArrival")
        cfg["queues"][qname] = entry

    for arc in model.get("network", []):
        cfg["network"].append({
            "source": arc["source"],
            "target": arc["target"],
            "probability": float(arc["probability"]),
        })

    rndnumbers = []
    seeds = model.get("seeds")
    per_seed = model.get("rndnumbersPerSeed")
    if seeds:
        for s in seeds:
            rng = random.Random(int(s))
            for _ in range(int(per_seed)):
                rndnumbers.append(rng.random())
    else:
        rndnumbers = [float(x) for x in model.get("rndnumbers", [])]

    return cfg, rndnumbers

def summarize(sim: NetworkSim):
    lines = []
    lines.append(f"Tempo global de simulação: {sim.total_time:.6f}")
    lines.append(f"Números aleatórios consumidos: {sim.rnd.total_used}")
    lines.append("")

    for name in sorted(sim.queues.keys()):
        q = sim.queues[name]
        T = sim.total_time if sim.total_time > 0 else 1.0

        all_states = sorted(q.state_time.keys())
        probs = {n: q.state_time[n] / T for n in all_states}

        util = q.busy_time / T / q.servers if q.servers > 0 else 0.0
        L = q.area_num_in_system / T
        Lq = q.area_num_in_queue / T

        lines.append(f"[{name}] s={q.servers}, cap_espera={'inf' if math.isinf(q.capacity) else q.capacity}")
        lines.append(f"  Perdas (bloqueios): {q.losses}")
        lines.append(f"  Completados: {q.completed} | Chegadas totais (ext+int): {q.arrived}")
        lines.append(f"  Utilização média por servidor: {util:.6f}")
        lines.append(f"  L (médio no sistema): {L:.6f} | Lq (médio na fila): {Lq:.6f}")
        lines.append("  Distribuição de estados (P[N=n]):")
        for n in all_states:
            lines.append(f"    n={n:2d} -> P={probs[n]:.6f} | tempo={q.state_time[n]:.6f}")
        lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(
        description="Simulador de rede de filas G/G/s/K (dirigido por YAML)."
    )
    ap.add_argument("model", help="Arquivo .yml do modelo (estilo Módulo 3).")
    ap.add_argument(
        "--stop-after",
        dest="stop_after",
        type=int,
        default=100000,
        help="Parar após consumir N aleatórios (default: 100000).",
    )
    args = ap.parse_args()

    cfg, rndnumbers = load_model(args.model)
    if not rndnumbers:
        raise SystemExit(
            "Nenhum aleatório disponível. Preencha 'seeds' + 'rndnumbersPerSeed' ou 'rndnumbers' no YAML."
        )

    rnd = RandomStream(rndnumbers)
    sim = NetworkSim(cfg, rnd, stop_after_randoms=args.stop_after)

    sim.run()
    print(summarize(sim))

if __name__ == "__main__":
    main()