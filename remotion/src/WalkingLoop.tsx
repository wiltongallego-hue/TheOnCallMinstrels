import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from 'remotion';
import timeline from '../public/timeline.json';

export type SpriteRef = {
  /** sprite image (RGBA crop of the wizard taken from one source frame) */
  src: string;
  /** top-left corner of the crop in the source frame */
  x: number;
  y: number;
  w: number;
  h: number;
  /** horizontal pivot used when mirroring (the wizard's body centre in the source frame) */
  pivotX: number;
  /** where the pivot must land in the output frame, plus vertical offset */
  targetX: number;
  dy: number;
  mirror: boolean;
};

export type TimelineEntry = {
  bg: string;
  sprite: SpriteRef | null;
};

const entries = timeline.frames as unknown as TimelineEntry[];
export const LOOP_FRAMES = entries.length;

const Sprite: React.FC<{s: SpriteRef}> = ({s}) => {
  // Place the crop so that its pivot lands on targetX; mirroring flips the crop around the pivot.
  const pivotInCrop = s.pivotX - s.x;
  const left = s.mirror ? s.targetX - (s.w - pivotInCrop) : s.targetX - pivotInCrop;
  return (
    <Img
      src={staticFile(s.src)}
      style={{
        position: 'absolute',
        left,
        top: s.y + s.dy,
        width: s.w,
        height: s.h,
        transform: s.mirror ? 'scaleX(-1)' : undefined,
        imageRendering: 'auto',
      }}
    />
  );
};

export const WalkingLoop: React.FC = () => {
  const frame = useCurrentFrame();
  const e = entries[frame % entries.length];
  return (
    <AbsoluteFill style={{backgroundColor: 'black'}}>
      <Img src={staticFile(e.bg)} style={{position: 'absolute', left: 0, top: 0, width: 1280, height: 720}} />
      {e.sprite ? <Sprite s={e.sprite} /> : null}
    </AbsoluteFill>
  );
};
