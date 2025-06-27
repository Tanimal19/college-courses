b11902038 資工三 鄭博允

---

# 1. 
Here's a sample flow between client and server when performing video streaming:
![|500](School/Course%20Homeworks/Computer%20Networks/assets/Pasted%20image%2020241119194858.png)

The detail is explained below:
1. Client send "GET" request to the video endpoint, and the server respond with an .html (or .rhtml) file which contiain a player (in the homework, we use [Shake Player](https://github.com/shaka-project/shaka-player), which is a javascript library)
2. After loading the .html file, the player (some javascript code) will start running in the browser, try to fetch the .mpd file. (by sending a "GET" request)
3. After getting the .mpd file, the player interpret the .mpd file and try to "GET" initial segment of the video. (the explanation of .mpd is at below)
4. After getting the initialize segment, which contains the metadata required to decoded the video, the player will decide which video segments to fetch by current status (for example, fetch lower quality video if bandwidth is limited).

Here's an example of .mpd file (some are omitted):
```xml
<?xml version="1.0" encoding="utf-8"?>
<MPD ...>
	...
	<Period id="0" start="PT0.0S">
		<AdaptationSet id="0" contentType="video" ...>
			<Representation id="0" mimeType="video/mp4"  codecs="avc1.42c00c" bandwidth="144000" width="256" height="144" frameRate="30000/1001">
				<SegmentTemplate timescale="30000" initialization="init-stream$RepresentationID$.m4s" media="chunk-stream$RepresentationID$-$Number%05d$.m4s" startNumber="1">
					<SegmentTimeline>
						<S t="0" d="240240" r="2" />
						<S d="161161" />
					</SegmentTimeline>
				</SegmentTemplate>
			</Representation>
			<Representation id="2" mimeType="video/mp4" codecs="avc1.640028" bandwidth="6000000" width="1920" height="1080" frameRate="30000/1001">
				<SegmentTemplate timescale="30000" initialization="init-stream$RepresentationID$.m4s" media="chunk-stream$RepresentationID$-$Number%05d$.m4s" startNumber="1">
					<SegmentTimeline>
						<S t="0" d="240240" r="2" />
						<S d="161161" />
					</SegmentTimeline>
				</SegmentTemplate>
			</Representation>
		</AdaptationSet>
		<AdaptationSet id="1" contentType="audio" ...>
			...
		</AdaptationSet>
	</Period>
</MPD>
```
- `AdaptationSet`: describe different media type, in the example, we have two media types: "video" and "audio"
- `Representation`: represent data with different resolution and framerate, in the example, the video has two resolutions: 256x144 and 1920x1080
- `SegmentTemplate`: mention the resource path, in the example, the initialize segment is "init-stream\$RepresentationID\$.m4s" and the video segments are "chunk-stream\$RepresentationID\$-\$Number%05d\$.m4s"
- `SegmentTimeline`: indicate the start time and end time of each segment

<div style="page-break-after:always;"></div>

# 2.

MP4 is a file format that contain video, audio (sometimes subtitles) all in one file. On the other hand, DASH split media into different streams (e.g. audio stream, video stream) and each stream to small chunks.

In conclusion, DASH is much resource efficient than MP4, both for client and server, but slightly complex to implement on an HTTP server:
1. Since DASH split media into small chunks, client don't need to buffer a large MP4 flie to provide smooth video playback.
2. Client can easily change the video quality or framerate by request a corresponding media stream, making it more adaptive. While if using MP4, client need to request a different MP4 file, which is large and need many time to transfer.
3. When a client request for a MP4 file, the server need to continous sending a large file, thus the server can't handle too much clients at the same time. When using DASH, server only need to send a small chunk, which is much faster.
4. To handle DASH, the server need to equip with video processing tools (e.g. [FFmpeg](https://www.ffmpeg.org/)), also the client need a tool to interpret .mpd file and video chunks. While MP4 is literally a file and easy to handle. However, nowadays, there's many libraries and tools which provide convenient way to server and client to handle DASH files. 

<div style="page-break-after:always;"></div>

# 3.
Authentication Flow:
1. Clinet send "GET" request to some endpoint that required authentication (e.g. "/upload/file")
2. Server respond with header like this:
	```
	HTTP/1.1 401 Unauthorized
	...
	WWW-Authenticate: Basic realm="B11902038"
	...
	```
3. Client recieve response from Server, and ask the user to input username and password (based on the format indicate by `WWW-Authenticate`).
   then send the same request to the same endpoint but adding `Authorization` header like this:
	```
	Authorization: Basic username1:password1
	```
	to be noted, the header will be encoded with base64, so the actual header will be like:
	```
	Authorization: Basic dXNlcm5hbWUxOnBhc3N3b3JkMQ==
	```
4. Server will decode the `Authorization` header with base64 decoder, then check if this username and password is valid.
   if valid, then Server will response with what Client ask for.
   if not, then go back to step 2.

Obviously, this authentication method is not secure, cause everyone can check the package, then get the username and password after decoding it (base64 is **very** easy to decode).

One alternative method can be using HTTPS, where the package will be encryption, which is **not** easy to decrypt.

<div style="page-break-after:always;"></div>

# Bonus: Git Commit History
## How did you utilize Git for development in this assignment?
- **git pull** whenever I start develop
- **git commit** whenever I finish some part, or some function
- **git push** whenever I finish today's work

## What benefits did it bring?
1. don't need to worry about modify the code too much and break it, because you can restore it any time (if you commit it)
2. actually, I have two devices to develop at home and at school, so using git (with github) is very helpful to sync my code between two devices. I can develop on one device and push my code, then pull my code if I want to develop on another device.
3. you can check your git commit history and realize how you messed up **everything**

## Is it a better way for you to submit homework via GitHub Classroom than via traditional ways (e.g., submit a .zip file to NTU Cool)? Why or Why not?
Submit a .zip file is the **dumbest** way I've seen in my entire life. I need to zip my code whenver I want to upload to NTU Cool, and if there's some minor fix, I need to zip whole thing again, which is really annoying. And you can't actually check what you've submit if you download the .zip and unzip it. 

On the otherhand, using GitHub Classroom is a better way since I can check what I've submit easily. Also, I can modify the code on Github when there's just some small bug, no need to zip-unzip again.

<div style="page-break-after:always;"></div>

# Reference
both for code and report
- https://book.itheima.net/course/223/1277519158031949826/1277529226395787267
- https://developer.mozilla.org/en-US/docs/Web/HTTP/Authentication
- https://stackoverflow.com/questions/11405819/does-struct-hostent-have-a-field-h-addr
- https://stackoverflow.com/questions/11405819/does-struct-hostent-have-a-field-h-addr
- https://stackoverflow.com/questions/49821687/how-to-determine-if-i-received-entire-message-from-recv-calls
- https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST
- https://zonble.github.io/understanding_audio_files/dash/
- ChatGPT (https://chatgpt.com/share/6727a5c4-1928-8012-b354-f05eec11cbd7)