from lupa.lua54 import LuaRuntime

lua = LuaRuntime(unpack_returned_tuples=True)
# lua.package.cpath = "C:\\Users\\Gebruiker\\Documents\\MEGA\\modding\\modded balatro"
# test = lua.eval("require('test')", name=db)
# print(test)
# lua.test[1].tester(1)
test = lua.eval(function(f,n) return )
lua.require('test2')
lua.execute('require test2 text2.tester(1)')
