import multiprocessing
from khrylib.rl.core import LoggerRL, TrajBatch
from khrylib.utils.memory import Memory
from khrylib.utils.torch import *
import math
import time
import os
import platform
if platform.system() != "Linux":
    from multiprocessing import set_start_method
    set_start_method("spawn")
os.environ["OMP_NUM_THREADS"] = "1"


class Agent:

    def __init__(self, env, policy_net, value_net, dtype, device, gamma,
                 num_threads=1, logger_cls=LoggerRL, logger_kwargs=None, traj_cls=TrajBatch):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.traj_cls = traj_cls
        self.logger_cls = logger_cls
        self.logger_kwargs = dict() if logger_kwargs is None else logger_kwargs
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]
        self.tf_objects = None  # Add storage for TF objects

    def __getstate__(self):
        """Custom state retrieval for pickling."""
        state = self.__dict__.copy()
        # Remove unpicklable TF objects
        if 'tf_objects' in state:
            del state['tf_objects']
        return state

    def __setstate__(self, state):
        """Custom state restoration for unpickling."""
        self.__dict__.update(state)
        self.tf_objects = None

    def initialize_tf(self):
        """Initialize TensorFlow objects lazily."""
        if self.tf_objects is None:
            # Initialize any TensorFlow objects here if needed
            pass

    def sample_worker(self, pid, queue, num_samples, mean_action):
        # Initialize TF objects in worker process
        self.initialize_tf()
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls(**self.logger_kwargs)

        while logger.num_steps < num_samples:
            state = self.env.reset()
            logger.start_episode(self.env)

            for t in range(10000):
                state_var = tensor(state).unsqueeze(0)
                trans_out = self.trans_policy(state_var)
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                action = self.policy_net.select_action(trans_out, use_mean_action)[0].numpy()
                action = int(action) if self.policy_net.type == 'discrete' else action.astype(np.float64)
                next_state, reward, done, info = self.env.step(action)
                logger.step(self.env, reward, info)

                mask = 0 if done else 1
                exp = 1 - use_mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp)

                if done:
                    break
                state = next_state

            logger.end_episode(self.env)

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger

    def sample(self, num_samples, mean_action=False, nthreads=None):
        if nthreads is None:
            nthreads = self.num_threads
        t_start = time.time()
        to_test(*self.sample_modules)
        
        # Use get_context to ensure proper multiprocessing context
        ctx = multiprocessing.get_context('spawn' if platform.system() != "Linux" else None)
        
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_num_samples = int(math.floor(num_samples / nthreads))
                queue = ctx.Queue()
                memories = [None] * nthreads
                loggers = [None] * nthreads
                
                # Start worker processes
                workers = []
                for i in range(nthreads-1):
                    worker_args = (i+1, queue, thread_num_samples, mean_action)
                    worker = ctx.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                    workers.append(worker)
                    
                # Run first worker in main process
                memories[0], loggers[0] = self.sample_worker(0, None, thread_num_samples, mean_action)

                # Collect results from other workers
                for i in range(nthreads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                
                # Wait for all workers to finish
                for worker in workers:
                    worker.join()
                
                traj_batch = self.traj_cls(memories)
                logger = self.logger_cls.merge(loggers, **self.logger_kwargs)

        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    def seed_worker(self, pid):
        if pid > 0:
            torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
            np.random.seed(np.random.randint(5000) * pid)

    def push_memory(self, memory, state, action, mask, next_state, reward, exp):
        memory.push(state, action, mask, next_state, reward, exp)

    def trans_policy(self, states):
        """transform states before going into policy net"""
        return states

    def trans_value(self, states):
        """transform states before going into value net"""
        return states

    def set_noise_rate(self, noise_rate):
        self.noise_rate = noise_rate