import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

from PIL import Image


@dataclass
class _Pending:
	text: str
	image: Image.Image
	event: threading.Event
	result: Optional[float] = None


class MicroBatcher:
	def __init__(self, max_batch_size: int, max_delay_ms: int, process_fn: Callable[[List[str], List[Image.Image]], List[float]]):
		self.max_batch_size = max_batch_size
		self.max_delay_ms = max_delay_ms
		self.process_fn = process_fn
		self._lock = threading.Lock()
		self._queue: List[_Pending] = []
		self._timer: Optional[threading.Timer] = None

	def submit(self, text: str, image: Image.Image) -> float:
		p = _Pending(text=text, image=image, event=threading.Event())
		with self._lock:
			self._queue.append(p)
			if len(self._queue) >= self.max_batch_size:
				self._flush_locked()
			else:
				self._ensure_timer_locked()
		# wait for completion
		p.event.wait()
		return float(p.result or 0.0)

	def _ensure_timer_locked(self) -> None:
		if self._timer is None or not self._timer.is_alive():
			self._timer = threading.Timer(self.max_delay_ms / 1000.0, self._try_flush)
			self._timer.start()

	def _try_flush(self) -> None:
		with self._lock:
			self._flush_locked()

	def _flush_locked(self) -> None:
		if not self._queue:
			return
		batch = self._queue[: self.max_batch_size]
		del self._queue[: self.max_batch_size]
		# cancel outstanding timer if queue emptied
		if not self._queue and self._timer is not None:
			self._timer.cancel()
			self._timer = None
		texts = [p.text for p in batch]
		images = [p.image for p in batch]
		scores = self.process_fn(texts, images)
		for p, s in zip(batch, scores):
			p.result = float(s)
			p.event.set()
