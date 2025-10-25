import threading
from collections import OrderedDict
from typing import Generic, MutableMapping, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
	def __init__(self, capacity: int = 1024) -> None:
		self.capacity = capacity
		self._store: MutableMapping[K, V] = OrderedDict()
		self._lock = threading.Lock()

	def get(self, key: K) -> Optional[V]:
		with self._lock:
			value = self._store.get(key)
			if value is not None:
				self._store.move_to_end(key)
			return value

	def put(self, key: K, value: V) -> None:
		with self._lock:
			if key in self._store:
				self._store.move_to_end(key)
			self._store[key] = value
			if len(self._store) > self.capacity:
				self._store.popitem(last=False)

	def __len__(self) -> int:
		with self._lock:
			return len(self._store)
