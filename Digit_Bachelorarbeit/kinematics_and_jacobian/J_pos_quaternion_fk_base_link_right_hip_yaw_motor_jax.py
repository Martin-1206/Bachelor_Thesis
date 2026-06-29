import jax.numpy as jnp
import numpy as np
import jax
@jax.jit
def J_pos_quaternion_fk_base_link_right_hip_yaw_motor(x):
	var1=0.101
	var2=0.694697
	var3=2
	var4=(x[16]/var3)
	var5=jnp.sin(var4)
	var6=0.131892
	var7=(x[16]/var3)
	var8=jnp.cos(var7)
	var9=((var2*var5)+(var6*var8))
	var10=0.5
	var11=(var10*jnp.cos(var4))
	var12=(var10*jnp.sin(var7))
	var13=((var2*var11)-(var6*var12))
	var14=-0.101
	var15=-0.694697
	var16=-0.131892
	var17=((var15*var8)-(var16*var5))
	var18=((var15*var12)+(var16*var11))
	var19=-0.088
	var20=((var2*var12)+(var6*var11))
	var21=((var2*var8)-(var6*var5))
	var22=0.088
	var23=((var16*var8)+(var15*var5))
	var24=((var15*var11)-(var16*var12))
	var25=-0.707107
	var26=(x[17]/var3)
	var27=jnp.cos(var26)
	var28=(var25*var27)
	var29=0.707107
	var30=(var29*var27)
	var31=(x[17]/var3)
	var32=jnp.sin(var31)
	var33=(var25*var32)
	var34=(var29*var32)
	var35=(var10*jnp.sin(var26))
	var36=(var25*var35)
	var37=(var29*var35)
	var38=(var10*jnp.cos(var31))
	var39=(var25*var38)
	var40=(var29*var38)
	return jnp.array([[((((((var1*((var9+var9)*var13))+(var14*((var17+var17)*var18)))+(var19*(var17*var20)))+(var19*(var21*var18)))+(var22*(var23*var13)))+(var22*(var9*var24))), 00], 
 [((((((((var14*(var21*var13))+(var1*(var9*var20)))+(var14*(var17*var24)))+(var1*(var23*var18)))+(var22*(var17*var13)))+(var19*(var9*var18)))+(var19*(var21*var24)))+(var22*(var23*var20))), 00], 
 [((((((var14*(var23*var13))+(var14*(var9*var24)))+(var14*(var17*var20)))+(var14*(var21*var18)))+(var19*((var23+var23)*var24)))+(var22*((var17+var17)*var18))), 00], 
 [((((var28*var18)-(var30*var20))-(var33*var24))-(var34*var13)), ((((var17*var36)-(var21*var37))-(var23*var39))-(var9*var40))], 
 [((((var30*var24)-(var33*var20))-(var34*var18))-(var28*var13)), ((((var21*var39)-(var23*var37))+(var17*var40))+(var9*var36))], 
 [((((var33*var13)-(var28*var20))-(var34*var24))-(var30*var18)), ((((var9*var39)-(var21*var36))-(var23*var40))-(var17*var37))], 
 [((((var28*var24)-(var34*var20))+(var33*var18))+(var30*var13)), ((((var21*var40)-(var23*var36))-(var17*var39))-(var9*var37))]])