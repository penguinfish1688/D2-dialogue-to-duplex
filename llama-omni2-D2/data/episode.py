from __future__ import annotations
from dataclasses import dataclass
from d2_llama.core.constants import (
    INPUT_SAMPLES_PER_FRAME,
    TTS_EOS_TOKEN_ID,
    TTS_SEPARATOR_TOKEN_ID,
    TTS_TEXT_END_TOKEN_ID,
)
from d2_llama.model.tts import build_read_write_schedule


@dataclass
class Episode:
    identity: str
    first_frame: int
    lexical: tuple[int, ...]
    audio_ids: tuple[int, ...]
    stop_frame: int | None = None

    def slots(self):
        """Native sequence plus causal issue clocks and explicit interrupt EOS.

        Each unit keeps its 40-ms acoustic clock and cannot precede the last
        condition needed by its Read-3 block. A condition already predicted
        by the Thinker may wait for that block's audio clock. No future
        Thinker hidden state enters a prior chunk or an interrupted response.
        At interruption, the control's own hidden state conditions native END,
        then SEP if needed, then EOS: no later speech target survives.
        """
        ids = self.lexical + (TTS_TEXT_END_TOKEN_ID,)
        clock = self.first_frame
        result = []
        for slot in build_read_write_schedule(len(ids), len(self.audio_ids)):
            condition_frame = -1
            label = -100
            if slot.kind == "condition":
                condition_frame = self.first_frame + slot.source_index
                clock = max(clock, condition_frame)
                token = ids[slot.source_index]
            elif slot.kind == "separator":
                token = TTS_SEPARATOR_TOKEN_ID
            else:
                clock = max(
                    clock, self.first_frame + slot.source_index * 640 // INPUT_SAMPLES_PER_FRAME
                )
                token = label = self.audio_ids[slot.source_index]
            if self.stop_frame is not None and clock >= self.stop_frame:
                break
            result.append((clock, slot.kind, token, condition_frame, label))
        if self.stop_frame is not None and not any(slot[4] == TTS_EOS_TOKEN_ID for slot in result):
            result.append(
                (
                    self.stop_frame,
                    "interrupt_condition",
                    TTS_TEXT_END_TOKEN_ID,
                    self.stop_frame,
                    -100,
                )
            )
            if not any(slot[1] == "separator" for slot in result):
                result.append((self.stop_frame, "separator", TTS_SEPARATOR_TOKEN_ID, -1, -100))
            result.append((self.stop_frame, "speech", TTS_EOS_TOKEN_ID, -1, TTS_EOS_TOKEN_ID))
        return tuple(result)
