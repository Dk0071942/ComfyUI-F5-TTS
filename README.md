
ComfyUI node to make text to speech audio with your own voice.

Using F5-TTS https://github.com/SWivid/F5-TTS

### Instructions

* Put in ComfyUI's "input" folder a .wav file of an audio of the voice you'd like to use, remove any background music, noise.
* And a .txt file of the same name with what was said.
* Press refresh to see it in the node

You can use the examples here...
* [Examples voices](examples/)
* [Simple workflow](examples/simple_ComfyUI_F5TTS_workflow.json)
* [Workflow with input audio only, using OpenAI's Whisper to get the text](examples/F5TTS_whisper_workflow.json)


ODE methods are documented at [torchdiffeq](https://github.com/rtqichen/torchdiffeq#keyword-arguments-for-odeint_adjoint)

### Multi voices...

Put your sample voice files into the "input" folder like...
```
voice.wav
voice.txt
voice.deep.wav
voice.deep.txt
voice.chipmunk.wav
voice.chipmunk.txt
```

Then you can use prompts for different voices...
```
{main} Hello World this is the end
{deep} This is the narrator
{chipmunk} Please, I need more helium
```


