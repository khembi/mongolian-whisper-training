/**
 * Remotion transcription example — copy into your Remotion project.
 *
 * Prerequisites:
 *   npm install @remotion/install-whisper-cpp @remotion/captions
 *   Place ggml-large-v3-mn.bin in whisper.cpp/models/
 *
 * Usage:
 *   npx tsx remotion/transcribe-mongolian.ts path/to/video.mp4
 */

import path from 'path';
import fs from 'fs';
import {execSync} from 'child_process';
import {
  installWhisperCpp,
  transcribe,
  toCaptions,
} from '@remotion/install-whisper-cpp';

const WHISPER_VERSION = '1.5.5';
const MODEL_FILE = 'ggml-large-v3-mn.bin';

const main = async () => {
  const inputPath = process.argv[2];
  if (!inputPath) {
    console.error('Usage: npx tsx remotion/transcribe-mongolian.ts <video-or-audio>');
    process.exit(1);
  }

  const whisperPath = path.join(process.cwd(), 'whisper.cpp');
  const modelPath = path.join(whisperPath, 'models', MODEL_FILE);

  if (!fs.existsSync(modelPath)) {
    console.error(`Model not found: ${modelPath}`);
    console.error('Copy your trained ggml-large-v3-mn.bin into whisper.cpp/models/');
    process.exit(1);
  }

  const wavPath = inputPath.replace(/\.[^.]+$/, '') + '.16k.wav';

  await installWhisperCpp({to: whisperPath, version: WHISPER_VERSION});

  console.log('Converting audio to 16kHz WAV...');
  execSync(
    `ffmpeg -i "${inputPath}" -ar 16000 -ac 1 "${wavPath}" -y`,
    {stdio: 'inherit'},
  );

  console.log('Transcribing (Mongolian, word-level timestamps)...');
  const whisperCppOutput = await transcribe({
    inputPath: wavPath,
    whisperPath,
    whisperCppVersion: WHISPER_VERSION,
    model: 'large-v3',
    modelFolder: path.join(whisperPath, 'models'),
    language: 'mn',
    tokenLevelTimestamps: true,
    splitOnWord: true,
  });

  const {captions} = toCaptions({whisperCppOutput});
  const outPath = inputPath.replace(/\.[^.]+$/, '') + '.captions.json';
  fs.writeFileSync(outPath, JSON.stringify(captions, null, 2));

  console.log(`Done. ${captions.length} caption tokens → ${outPath}`);
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
