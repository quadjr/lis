﻿using UnityEngine;
using System;
using System.Collections.Generic;
using WebSocketSharp;
using WebSocketSharp.Server;
using WebSocketSharp.Net;
using System.Threading;
using MsgPack;

namespace MLPlayer
{
	public class AIServer : MonoBehaviour
	{
		private WebSocketServer wssv;
		                                          
		public Queue<byte[]> agentMessageQueue;
		private Queue<byte[]> aiMessageQueue;
		private Mutex mutAgent;
		public Agent agent;
		private MsgPack.CompiledPacker packer;

		public AIServer (Agent _agent)
		{
			agent = _agent;
			mutAgent = new Mutex ();
			packer = new MsgPack.CompiledPacker ();
			agentMessageQueue = new Queue<byte[]> ();
			aiMessageQueue = new Queue<byte[]> ();
		}

		public class CommunicationGym : WebSocketBehavior
		{
			public Agent agent { set; get; }
			MsgPack.BoxingPacker packer = new MsgPack.BoxingPacker ();
			private bool SendFlag=false;

			protected override void OnMessage (MessageEventArgs e)
			{
				//receive message 
				agent.action.Set ((Dictionary<System.Object,System.Object>)packer.Unpack (e.RawData));
				SceneController.received.Set ();
				Debug.Log ("Rotate=" + agent.action.rotate + " Forword=" + agent.action.forward + " Jump=" + agent.action.jump);
				SendFlag = false;

				//send state data 
				Sendmessage();
			}

			protected override void OnOpen ()
			{
				Debug.Log ("Socket Open");

				SceneController.received.Set ();
				Sendmessage ();
			}

			private void Sendmessage(){
				SendFlag = false;
				//send state data 
				while (SendFlag == false) {
					if (SceneController.server.agentMessageQueue.Count > 0) {
						byte[] data = SceneController.server.PopAgentState ();
						Send (data);
						SendFlag = true;
					}
				}
			}
				
		}

		CommunicationGym instantiate ()
		{
			CommunicationGym service = new CommunicationGym ();
			service.agent = agent;
			return service;
		}

		void Awake ()
		{
			wssv = new WebSocketServer ("ws://localhost:" + 4649);
			wssv.AddWebSocketService<CommunicationGym> ("/CommunicationGym", instantiate);
			wssv.Start ();


			if (wssv.IsListening) {
				Debug.Log ("Listening on port " + wssv.Port + ", and providing WebSocket services:");
				foreach (var path in wssv.WebSocketServices.Paths)
					Debug.Log ("- " + path);
			}
		}

		public void PushAIMessage (byte[] msg)
		{
			throw new System.NotImplementedException ();
		}

		public byte[] PopAIMessage ()
		{
			throw new System.NotImplementedException ();
		}

		public void PushAgentState (State s)
		{
			byte[] msg = packer.Pack (s);  
			mutAgent.WaitOne ();
			agentMessageQueue.Enqueue (msg);  
			mutAgent.ReleaseMutex ();
		}

		public byte[] PopAgentState ()
		{
			byte[] received = null;

			mutAgent.WaitOne ();
			if (agentMessageQueue.Count > 0) {
				received = agentMessageQueue.Dequeue ();
			}
			mutAgent.ReleaseMutex ();

			return received;
		}

		void OnApplicationQuit ()
		{
			wssv.Stop ();
			Debug.Log ("websocket server exiteed");
		}
	}
}
