import Head from 'next/head';
import { useEffect } from 'react';

export default function Home(){
  useEffect(()=>{
    const s = document.createElement('script');
    s.src = '/widget.js';
    s.onload = ()=> {
      window.MultilingualWidget.init({ tokenEndpoint: '/api/session', userId: 'demo-user', wsUrl: 'ws://localhost:8000' });
    };
    document.body.appendChild(s);
    return ()=> { document.body.removeChild(s); };
  }, []);
  return (
    <>
      <Head><title>Integrator Demo</title></Head>
      <div style={{ padding: 24 }}>
        <h1>Integrator Demo</h1>
        <p>Floating Voice AI widget will appear in the bottom-right.</p>
      </div>
    </>
  )
}
