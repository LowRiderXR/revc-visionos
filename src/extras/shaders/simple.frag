uniform sampler2D tex0;

FSIN vec4 v_color;
FSIN vec2 v_tex0;
FSIN float v_fog;

void
main(void)
{
	vec4 color;
// Mirrors the identical block in librw's shaders/simple.frag: the ped texture atlas bleeds
// at low pixel coverage, so the visionOS build compiles the ped shaders with a negative mip
// bias. reVC's neo RIM pipeline uses THIS copy of the shader (custompipes_gl.cpp), and peds
// get that pipeline unconditionally (CPedModelInfo::SetClump -> AttachRimPipe), so without
// the block here the fix silently stops applying whenever NeoRimLight is on.
#ifdef VC_SKIN_LODBIAS
	color = v_color*texture(tex0, vec2(v_tex0.x, 1.0-v_tex0.y), float(VC_SKIN_LODBIAS));
#else
	color = v_color*texture(tex0, vec2(v_tex0.x, 1.0-v_tex0.y));
#endif
	color.rgb = mix(u_fogColor.rgb, color.rgb, v_fog);
	DoAlphaTest(color.a);

	FRAGCOLOR(color);
}

