"""Official Colab MCP via official Python MCP SDK; no browser-cookie access."""
import asyncio,contextlib,json,os,sys,time
from datetime import timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CACHE=ROOT/'data/.colab_connection'
def save(p,z):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(z,indent=2),encoding='utf8');tmp.replace(p)
async def main():
    from mcp import ClientSession,StdioServerParameters
    from mcp.client.stdio import stdio_client
    CACHE.mkdir(exist_ok=True);requests=CACHE/'requests';requests.mkdir(exist_ok=True)
    server=ROOT/'data/.colab_mcp_env/Scripts/colab-mcp.exe';assert server.exists()
    env={k:v for k,v in os.environ.items() if k.upper() in {'PATH','SYSTEMROOT','WINDIR','COMSPEC','USERPROFILE','TEMP','TMP','APPDATA','LOCALAPPDATA','PROGRAMFILES','PROGRAMFILES(X86)'}}
    state={'pid':os.getpid(),'started':time.time(),'server':'googlecolab/colab-mcp','official_commit':'b9ab3899e0f1fa493390b1fd6d54aa2e464ecdf1','status':'starting','connected':False}
    async def heartbeat():
        while True:save(CACHE/'status.json',{**state,'updated':time.time()});await asyncio.sleep(3)
    beat=asyncio.create_task(heartbeat())
    try:
        with (CACHE/'server.stderr.log').open('a',encoding='utf8') as err:
            async with stdio_client(StdioServerParameters(command=str(server),env=env,cwd=str(ROOT)),errlog=err) as (read,write):
                async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=180)) as session:
                    await session.initialize();state['status']='ready'
                    listing=await session.list_tools();save(CACHE/'tools.json',listing.model_dump(mode='json'));state['tools']=[x.name for x in listing.tools]
                    print('Official Colab MCP ready; browser approval is required for connection.',flush=True)
                    while True:
                        for p in sorted(requests.glob('*.json')):
                            result=p.with_suffix('.result.json')
                            if p.name.endswith('.result.json') or result.exists():continue
                            task=json.loads(p.read_text());state['status']='busy';state['action']=task['action'];save(CACHE/'active.json',{'request':p.name,'action':task['action'],'started':time.time()})
                            try:
                                if task['action']=='list':
                                    r=await session.list_tools();save(CACHE/'tools.json',r.model_dump(mode='json'));state['tools']=[x.name for x in r.tools]
                                elif task['action']=='call':
                                    assert task['name'] in state['tools'],'Unknown tool; refresh list first'
                                    r=await session.call_tool(task['name'],arguments=task.get('arguments',{}))
                                    if task['name']=='open_colab_browser_connection':state['connected']=bool((r.structuredContent or {}).get('result',False))
                                else:raise ValueError('Unsupported bridge action')
                                save(result,{'ok':True,'time':time.time(),'response':r.model_dump(mode='json')})
                            except Exception as error:save(result,{'ok':False,'time':time.time(),'error':repr(error)})
                            state['status']='ready';state.pop('action',None)
                        await asyncio.sleep(.25)
    except BaseException as error:
        state.update(status='stopped',error=repr(error));save(CACHE/'status.json',{**state,'updated':time.time()});raise
    finally:beat.cancel()
if __name__=='__main__':
    CACHE.mkdir(parents=True,exist_ok=True)
    with (CACHE/'bridge.stdout.log').open('a',encoding='utf8') as out,(CACHE/'bridge.stderr.log').open('a',encoding='utf8') as err:
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):asyncio.run(main())
