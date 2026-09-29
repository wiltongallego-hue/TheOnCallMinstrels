import React from 'react';
import {Composition} from 'remotion';
import {LOOP_FRAMES, WalkingLoop} from './WalkingLoop';

export const RemotionRoot: React.FC = () => (
  <>
    {/* One full loop: the last frame flows straight back into the first. */}
    <Composition
      id="WalkingLoop"
      component={WalkingLoop}
      durationInFrames={LOOP_FRAMES}
      fps={24}
      width={1280}
      height={720}
    />
    {/* Three loops back to back, handy for checking the seam. */}
    <Composition
      id="WalkingLoopPreview3x"
      component={WalkingLoop}
      durationInFrames={LOOP_FRAMES * 3}
      fps={24}
      width={1280}
      height={720}
    />
  </>
);
