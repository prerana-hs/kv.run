import io

files = ['build/router/client/src/client.rs',
         'build/router/client/src/sharded_client.rs']

for file in files:
    with open(file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('mod kvr;\n' + content)
