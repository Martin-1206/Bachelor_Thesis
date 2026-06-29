import jax.numpy as jnp
import numpy as np
import jax
@jax.jit
def J_pos_ypr_fk_base_link_right_hip_yaw_motor(x):
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
	var25=1
	var26=-0.707107
	var27=(x[17]/var3)
	var28=jnp.cos(var27)
	var29=(var26*var28)
	var30=0.707107
	var31=(x[17]/var3)
	var32=jnp.sin(var31)
	var33=(var30*var32)
	var34=(var30*var28)
	var35=(var26*var32)
	var36=((((var21*var29)-(var23*var33))+(var17*var34))+(var9*var35))
	var37=((((var21*var33)+(var23*var29))-(var17*var35))+(var9*var34))
	var38=(var25-(var3*(jnp.square(var36)+jnp.square(var37))))
	var39=((((var21*var34)-(var23*var35))-(var17*var29))-(var9*var33))
	var40=((((var21*var35)+(var23*var34))+(var17*var33))-(var9*var29))
	var41=(var3*((var39*var37)+(var40*var36)))
	var42=(jnp.square(var41)+jnp.square(var38))
	var43=(var38/var42)
	var44=(((var29*var18)-((var34*var20)+(var35*var24)))-(var33*var13))
	var45=((((var29*var24)-(var33*var20))+(var35*var18))+(var34*var13))
	var46=((((var34*var24)-(var35*var20))-(var33*var18))-(var29*var13))
	var47=((var35*var13)-(((var29*var20)+(var33*var24))+(var34*var18)))
	var48=(var41/var42)
	var49=(var36+var36)
	var50=(var37+var37)
	var51=jnp.sqrt((var25-jnp.square((var3*((var39*var36)-(var37*var40))))))
	var52=(var25-(var3*(jnp.square(var40)+jnp.square(var36))))
	var53=(var3*((var39*var40)+(var36*var37)))
	var54=(jnp.square(var53)+jnp.square(var52))
	var55=(var52/var54)
	var56=(var53/var54)
	var57=(var40+var40)
	var58=(var36+var36)
	var59=(var10*jnp.sin(var27))
	var60=(var26*var59)
	var61=(var30*var59)
	var62=(var10*jnp.cos(var31))
	var63=(var26*var62)
	var64=(var30*var62)
	var65=(((var17*var60)-((var21*var61)+(var23*var63)))-(var9*var64))
	var66=((((var21*var64)-(var23*var60))-(var17*var63))-(var9*var61))
	var67=((((var21*var63)-(var23*var61))+(var17*var64))+(var9*var60))
	var68=((var9*var63)-(((var21*var60)+(var23*var64))+(var17*var61)))
	return jnp.array([[((((((var1*((var9+var9)*var13))+(var14*((var17+var17)*var18)))+(var19*(var17*var20)))+(var19*(var21*var18)))+(var22*(var23*var13)))+(var22*(var9*var24))), 00], 
 [((((((((var14*(var21*var13))+(var1*(var9*var20)))+(var14*(var17*var24)))+(var1*(var23*var18)))+(var22*(var17*var13)))+(var19*(var9*var18)))+(var19*(var21*var24)))+(var22*(var23*var20))), 00], 
 [((((((var14*(var23*var13))+(var14*(var9*var24)))+(var14*(var17*var20)))+(var14*(var21*var18)))+(var19*((var23+var23)*var24)))+(var22*((var17+var17)*var18))), 00], 
 [((var43*(var3*(((var37*var44)+(var39*var45))+((var36*var46)+(var40*var47)))))+(var48*(var3*((var49*var47)+(var50*var45))))), ((var43*(var3*(((var37*var65)+(var39*var66))+((var36*var67)+(var40*var68)))))+(var48*(var3*((var49*var68)+(var50*var66)))))], 
 [((var3*(((var36*var44)+(var39*var47))-((var40*var45)+(var37*var46))))/var51), ((var3*(((var36*var65)+(var39*var68))-((var40*var66)+(var37*var67))))/var51)], 
 [((var55*(var3*(((var40*var44)+(var39*var46))+((var37*var47)+(var36*var45)))))+(var56*(var3*((var57*var46)+(var58*var47))))), ((var55*(var3*(((var40*var65)+(var39*var67))+((var37*var68)+(var36*var66)))))+(var56*(var3*((var57*var67)+(var58*var68)))))]])