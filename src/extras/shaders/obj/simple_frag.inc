const char *simple_frag_src =
"uniform sampler2D tex0;\n"

"FSIN vec4 v_color;\n"
"FSIN vec2 v_tex0;\n"
"FSIN float v_fog;\n"

"void\n"
"main(void)\n"
"{\n"
"	vec4 color;\n"
"// Mirrors the identical block in librw's shaders/simple.frag: the ped texture atlas bleeds\n"
"// at low pixel coverage, so the visionOS build compiles the ped shaders with a negative mip\n"
"// bias. reVC's neo RIM pipeline uses THIS copy of the shader (custompipes_gl.cpp), and peds\n"
"// get that pipeline unconditionally (CPedModelInfo::SetClump -> AttachRimPipe), so without\n"
"// the block here the fix silently stops applying whenever NeoRimLight is on.\n"
"#ifdef VC_SKIN_LODBIAS\n"
"	color = v_color*texture(tex0, vec2(v_tex0.x, 1.0-v_tex0.y), float(VC_SKIN_LODBIAS));\n"
"#else\n"
"	color = v_color*texture(tex0, vec2(v_tex0.x, 1.0-v_tex0.y));\n"
"#endif\n"
"	color.rgb = mix(u_fogColor.rgb, color.rgb, v_fog);\n"
"	DoAlphaTest(color.a);\n"

"	FRAGCOLOR(color);\n"
"}\n"

;
