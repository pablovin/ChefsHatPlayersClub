��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8ߵ
x
Dense0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*
shared_nameDense0/kernel
q
!Dense0/kernel/Read/ReadVariableOpReadVariableOpDense0/kernel* 
_output_shapes
:
��*
dtype0
o
Dense0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameDense0/bias
h
Dense0/bias/Read/ReadVariableOpReadVariableOpDense0/bias*
_output_shapes	
:�*
dtype0
{
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�* 
shared_namedense_38/kernel
t
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes
:	�*
dtype0
r
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
 

0
1
2
3
 

0
1
2
3
�
trainable_variables
layer_regularization_losses
non_trainable_variables
regularization_losses
	variables
layer_metrics
metrics

layers
 
YW
VARIABLE_VALUEDense0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEDense0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
 layer_regularization_losses
!non_trainable_variables
regularization_losses
	variables
"layer_metrics
#metrics

$layers
 
 
 
�
trainable_variables
%layer_regularization_losses
&non_trainable_variables
regularization_losses
	variables
'layer_metrics
(metrics

)layers
[Y
VARIABLE_VALUEdense_38/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_38/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
trainable_variables
*layer_regularization_losses
+non_trainable_variables
regularization_losses
	variables
,layer_metrics
-metrics

.layers
 
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_RewardInputPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_RewardInputDense0/kernelDense0/biasdense_38/kerneldense_38/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_6802372
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!Dense0/kernel/Read/ReadVariableOpDense0/bias/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_6802518
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense0/kernelDense0/biasdense_38/kerneldense_38/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_6802540��
�	
�
E__inference_dense_38_layer_call_and_return_conditional_losses_6802268

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_model_62_layer_call_and_return_conditional_losses_6802390

inputs)
%dense0_matmul_readvariableop_resource*
&dense0_biasadd_readvariableop_resource+
'dense_38_matmul_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource
identity��Dense0/BiasAdd/ReadVariableOp�Dense0/MatMul/ReadVariableOp�dense_38/BiasAdd/ReadVariableOp�dense_38/MatMul/ReadVariableOp�
Dense0/MatMul/ReadVariableOpReadVariableOp%dense0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
Dense0/MatMul/ReadVariableOp�
Dense0/MatMulMatMulinputs$Dense0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Dense0/MatMul�
Dense0/BiasAdd/ReadVariableOpReadVariableOp&dense0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
Dense0/BiasAdd/ReadVariableOp�
Dense0/BiasAddBiasAddDense0/MatMul:product:0%Dense0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Dense0/BiasAdd�
leaky_re_lu_64/LeakyRelu	LeakyReluDense0/BiasAdd:output:0*(
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_64/LeakyRelu�
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_38/MatMul/ReadVariableOp�
dense_38/MatMulMatMul&leaky_re_lu_64/LeakyRelu:activations:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_38/MatMul�
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_38/BiasAdd/ReadVariableOp�
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_38/BiasAdds
dense_38/TanhTanhdense_38/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_38/Tanh�
IdentityIdentitydense_38/Tanh:y:0^Dense0/BiasAdd/ReadVariableOp^Dense0/MatMul/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2>
Dense0/BiasAdd/ReadVariableOpDense0/BiasAdd/ReadVariableOp2<
Dense0/MatMul/ReadVariableOpDense0/MatMul/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_6802372
rewardinput
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrewardinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_68022142
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_nameRewardInput
�
�
E__inference_model_62_layer_call_and_return_conditional_losses_6802300
rewardinput
dense0_6802288
dense0_6802290
dense_38_6802294
dense_38_6802296
identity��Dense0/StatefulPartitionedCall� dense_38/StatefulPartitionedCall�
Dense0/StatefulPartitionedCallStatefulPartitionedCallrewardinputdense0_6802288dense0_6802290*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_Dense0_layer_call_and_return_conditional_losses_68022282 
Dense0/StatefulPartitionedCall�
leaky_re_lu_64/PartitionedCallPartitionedCall'Dense0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_68022492 
leaky_re_lu_64/PartitionedCall�
 dense_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_64/PartitionedCall:output:0dense_38_6802294dense_38_6802296*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_68022682"
 dense_38/StatefulPartitionedCall�
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^Dense0/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2@
Dense0/StatefulPartitionedCallDense0/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_nameRewardInput
�

*__inference_dense_38_layer_call_fn_6802483

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_68022682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_model_62_layer_call_and_return_conditional_losses_6802408

inputs)
%dense0_matmul_readvariableop_resource*
&dense0_biasadd_readvariableop_resource+
'dense_38_matmul_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource
identity��Dense0/BiasAdd/ReadVariableOp�Dense0/MatMul/ReadVariableOp�dense_38/BiasAdd/ReadVariableOp�dense_38/MatMul/ReadVariableOp�
Dense0/MatMul/ReadVariableOpReadVariableOp%dense0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
Dense0/MatMul/ReadVariableOp�
Dense0/MatMulMatMulinputs$Dense0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Dense0/MatMul�
Dense0/BiasAdd/ReadVariableOpReadVariableOp&dense0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
Dense0/BiasAdd/ReadVariableOp�
Dense0/BiasAddBiasAddDense0/MatMul:product:0%Dense0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Dense0/BiasAdd�
leaky_re_lu_64/LeakyRelu	LeakyReluDense0/BiasAdd:output:0*(
_output_shapes
:����������*
alpha%���>2
leaky_re_lu_64/LeakyRelu�
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02 
dense_38/MatMul/ReadVariableOp�
dense_38/MatMulMatMul&leaky_re_lu_64/LeakyRelu:activations:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_38/MatMul�
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_38/BiasAdd/ReadVariableOp�
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_38/BiasAdds
dense_38/TanhTanhdense_38/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_38/Tanh�
IdentityIdentitydense_38/Tanh:y:0^Dense0/BiasAdd/ReadVariableOp^Dense0/MatMul/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2>
Dense0/BiasAdd/ReadVariableOpDense0/BiasAdd/ReadVariableOp2<
Dense0/MatMul/ReadVariableOpDense0/MatMul/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_model_62_layer_call_fn_6802421

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_62_layer_call_and_return_conditional_losses_68023182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_model_62_layer_call_fn_6802357
rewardinput
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrewardinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_62_layer_call_and_return_conditional_losses_68023462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_nameRewardInput
�
�
 __inference__traced_save_6802518
file_prefix,
(savev2_dense0_kernel_read_readvariableop*
&savev2_dense0_bias_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense0_kernel_read_readvariableop&savev2_dense0_bias_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*;
_input_shapes*
(: :
��:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: 
�
�
E__inference_model_62_layer_call_and_return_conditional_losses_6802285
rewardinput
dense0_6802239
dense0_6802241
dense_38_6802279
dense_38_6802281
identity��Dense0/StatefulPartitionedCall� dense_38/StatefulPartitionedCall�
Dense0/StatefulPartitionedCallStatefulPartitionedCallrewardinputdense0_6802239dense0_6802241*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_Dense0_layer_call_and_return_conditional_losses_68022282 
Dense0/StatefulPartitionedCall�
leaky_re_lu_64/PartitionedCallPartitionedCall'Dense0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_68022492 
leaky_re_lu_64/PartitionedCall�
 dense_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_64/PartitionedCall:output:0dense_38_6802279dense_38_6802281*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_68022682"
 dense_38/StatefulPartitionedCall�
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^Dense0/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2@
Dense0/StatefulPartitionedCallDense0/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_nameRewardInput
�	
�
E__inference_dense_38_layer_call_and_return_conditional_losses_6802474

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_Dense0_layer_call_and_return_conditional_losses_6802228

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
L
0__inference_leaky_re_lu_64_layer_call_fn_6802463

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_68022492
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
"__inference__wrapped_model_6802214
rewardinput2
.model_62_dense0_matmul_readvariableop_resource3
/model_62_dense0_biasadd_readvariableop_resource4
0model_62_dense_38_matmul_readvariableop_resource5
1model_62_dense_38_biasadd_readvariableop_resource
identity��&model_62/Dense0/BiasAdd/ReadVariableOp�%model_62/Dense0/MatMul/ReadVariableOp�(model_62/dense_38/BiasAdd/ReadVariableOp�'model_62/dense_38/MatMul/ReadVariableOp�
%model_62/Dense0/MatMul/ReadVariableOpReadVariableOp.model_62_dense0_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02'
%model_62/Dense0/MatMul/ReadVariableOp�
model_62/Dense0/MatMulMatMulrewardinput-model_62/Dense0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_62/Dense0/MatMul�
&model_62/Dense0/BiasAdd/ReadVariableOpReadVariableOp/model_62_dense0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02(
&model_62/Dense0/BiasAdd/ReadVariableOp�
model_62/Dense0/BiasAddBiasAdd model_62/Dense0/MatMul:product:0.model_62/Dense0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
model_62/Dense0/BiasAdd�
!model_62/leaky_re_lu_64/LeakyRelu	LeakyRelu model_62/Dense0/BiasAdd:output:0*(
_output_shapes
:����������*
alpha%���>2#
!model_62/leaky_re_lu_64/LeakyRelu�
'model_62/dense_38/MatMul/ReadVariableOpReadVariableOp0model_62_dense_38_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02)
'model_62/dense_38/MatMul/ReadVariableOp�
model_62/dense_38/MatMulMatMul/model_62/leaky_re_lu_64/LeakyRelu:activations:0/model_62/dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_62/dense_38/MatMul�
(model_62/dense_38/BiasAdd/ReadVariableOpReadVariableOp1model_62_dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_62/dense_38/BiasAdd/ReadVariableOp�
model_62/dense_38/BiasAddBiasAdd"model_62/dense_38/MatMul:product:00model_62/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_62/dense_38/BiasAdd�
model_62/dense_38/TanhTanh"model_62/dense_38/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
model_62/dense_38/Tanh�
IdentityIdentitymodel_62/dense_38/Tanh:y:0'^model_62/Dense0/BiasAdd/ReadVariableOp&^model_62/Dense0/MatMul/ReadVariableOp)^model_62/dense_38/BiasAdd/ReadVariableOp(^model_62/dense_38/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2P
&model_62/Dense0/BiasAdd/ReadVariableOp&model_62/Dense0/BiasAdd/ReadVariableOp2N
%model_62/Dense0/MatMul/ReadVariableOp%model_62/Dense0/MatMul/ReadVariableOp2T
(model_62/dense_38/BiasAdd/ReadVariableOp(model_62/dense_38/BiasAdd/ReadVariableOp2R
'model_62/dense_38/MatMul/ReadVariableOp'model_62/dense_38/MatMul/ReadVariableOp:U Q
(
_output_shapes
:����������
%
_user_specified_nameRewardInput
�
�
#__inference__traced_restore_6802540
file_prefix"
assignvariableop_dense0_kernel"
assignvariableop_1_dense0_bias&
"assignvariableop_2_dense_38_kernel$
 assignvariableop_3_dense_38_bias

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_dense0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_38_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_38_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
*__inference_model_62_layer_call_fn_6802434

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_62_layer_call_and_return_conditional_losses_68023462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
}
(__inference_Dense0_layer_call_fn_6802453

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_Dense0_layer_call_and_return_conditional_losses_68022282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
C__inference_Dense0_layer_call_and_return_conditional_losses_6802444

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_6802249

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������*
alpha%���>2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_model_62_layer_call_and_return_conditional_losses_6802346

inputs
dense0_6802334
dense0_6802336
dense_38_6802340
dense_38_6802342
identity��Dense0/StatefulPartitionedCall� dense_38/StatefulPartitionedCall�
Dense0/StatefulPartitionedCallStatefulPartitionedCallinputsdense0_6802334dense0_6802336*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_Dense0_layer_call_and_return_conditional_losses_68022282 
Dense0/StatefulPartitionedCall�
leaky_re_lu_64/PartitionedCallPartitionedCall'Dense0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_68022492 
leaky_re_lu_64/PartitionedCall�
 dense_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_64/PartitionedCall:output:0dense_38_6802340dense_38_6802342*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_68022682"
 dense_38/StatefulPartitionedCall�
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^Dense0/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2@
Dense0/StatefulPartitionedCallDense0/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
g
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_6802458

inputs
identitye
	LeakyRelu	LeakyReluinputs*(
_output_shapes
:����������*
alpha%���>2
	LeakyRelul
IdentityIdentityLeakyRelu:activations:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_model_62_layer_call_and_return_conditional_losses_6802318

inputs
dense0_6802306
dense0_6802308
dense_38_6802312
dense_38_6802314
identity��Dense0/StatefulPartitionedCall� dense_38/StatefulPartitionedCall�
Dense0/StatefulPartitionedCallStatefulPartitionedCallinputsdense0_6802306dense0_6802308*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_Dense0_layer_call_and_return_conditional_losses_68022282 
Dense0/StatefulPartitionedCall�
leaky_re_lu_64/PartitionedCallPartitionedCall'Dense0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_68022492 
leaky_re_lu_64/PartitionedCall�
 dense_38/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_64/PartitionedCall:output:0dense_38_6802312dense_38_6802314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_68022682"
 dense_38/StatefulPartitionedCall�
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^Dense0/StatefulPartitionedCall!^dense_38/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::2@
Dense0/StatefulPartitionedCallDense0/StatefulPartitionedCall2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_model_62_layer_call_fn_6802329
rewardinput
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallrewardinputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_62_layer_call_and_return_conditional_losses_68023182
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:����������::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
(
_output_shapes
:����������
%
_user_specified_nameRewardInput"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
RewardInput5
serving_default_RewardInput:0����������<
dense_380
StatefulPartitionedCall:0���������tensorflow/serving/predict:�u
�!
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
/__call__
0_default_save_signature
*1&call_and_return_all_conditional_losses"�
_tf_keras_network�{"class_name": "Functional", "name": "model_62", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_62", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 228]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RewardInput"}, "name": "RewardInput", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense0", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense0", "inbound_nodes": [[["RewardInput", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_64", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_64", "inbound_nodes": [[["Dense0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_38", "inbound_nodes": [[["leaky_re_lu_64", 0, 0, {}]]]}], "input_layers": [["RewardInput", 0, 0]], "output_layers": [["dense_38", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 228]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 228]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_62", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 228]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RewardInput"}, "name": "RewardInput", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "Dense0", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense0", "inbound_nodes": [[["RewardInput", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_64", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_re_lu_64", "inbound_nodes": [[["Dense0", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_38", "inbound_nodes": [[["leaky_re_lu_64", 0, 0, {}]]]}], "input_layers": [["RewardInput", 0, 0]], "output_layers": [["dense_38", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["mse"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "RewardInput", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 228]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 228]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "RewardInput"}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "Dense0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Dense0", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 228}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 228]}}
�
trainable_variables
regularization_losses
	variables
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_64", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_64", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
6__call__
*7&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
trainable_variables
layer_regularization_losses
non_trainable_variables
regularization_losses
	variables
layer_metrics
metrics

layers
/__call__
0_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
,
8serving_default"
signature_map
!:
��2Dense0/kernel
:�2Dense0/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
 layer_regularization_losses
!non_trainable_variables
regularization_losses
	variables
"layer_metrics
#metrics

$layers
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
trainable_variables
%layer_regularization_losses
&non_trainable_variables
regularization_losses
	variables
'layer_metrics
(metrics

)layers
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
": 	�2dense_38/kernel
:2dense_38/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
trainable_variables
*layer_regularization_losses
+non_trainable_variables
regularization_losses
	variables
,layer_metrics
-metrics

.layers
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
*__inference_model_62_layer_call_fn_6802421
*__inference_model_62_layer_call_fn_6802357
*__inference_model_62_layer_call_fn_6802434
*__inference_model_62_layer_call_fn_6802329�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_6802214�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *+�(
&�#
RewardInput����������
�2�
E__inference_model_62_layer_call_and_return_conditional_losses_6802408
E__inference_model_62_layer_call_and_return_conditional_losses_6802390
E__inference_model_62_layer_call_and_return_conditional_losses_6802285
E__inference_model_62_layer_call_and_return_conditional_losses_6802300�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_Dense0_layer_call_fn_6802453�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_Dense0_layer_call_and_return_conditional_losses_6802444�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
0__inference_leaky_re_lu_64_layer_call_fn_6802463�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_6802458�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_dense_38_layer_call_fn_6802483�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_38_layer_call_and_return_conditional_losses_6802474�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_6802372RewardInput"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
C__inference_Dense0_layer_call_and_return_conditional_losses_6802444^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� }
(__inference_Dense0_layer_call_fn_6802453Q0�-
&�#
!�
inputs����������
� "������������
"__inference__wrapped_model_6802214r5�2
+�(
&�#
RewardInput����������
� "3�0
.
dense_38"�
dense_38����������
E__inference_dense_38_layer_call_and_return_conditional_losses_6802474]0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� ~
*__inference_dense_38_layer_call_fn_6802483P0�-
&�#
!�
inputs����������
� "�����������
K__inference_leaky_re_lu_64_layer_call_and_return_conditional_losses_6802458Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
0__inference_leaky_re_lu_64_layer_call_fn_6802463M0�-
&�#
!�
inputs����������
� "������������
E__inference_model_62_layer_call_and_return_conditional_losses_6802285l=�:
3�0
&�#
RewardInput����������
p

 
� "%�"
�
0���������
� �
E__inference_model_62_layer_call_and_return_conditional_losses_6802300l=�:
3�0
&�#
RewardInput����������
p 

 
� "%�"
�
0���������
� �
E__inference_model_62_layer_call_and_return_conditional_losses_6802390g8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������
� �
E__inference_model_62_layer_call_and_return_conditional_losses_6802408g8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������
� �
*__inference_model_62_layer_call_fn_6802329_=�:
3�0
&�#
RewardInput����������
p

 
� "�����������
*__inference_model_62_layer_call_fn_6802357_=�:
3�0
&�#
RewardInput����������
p 

 
� "�����������
*__inference_model_62_layer_call_fn_6802421Z8�5
.�+
!�
inputs����������
p

 
� "�����������
*__inference_model_62_layer_call_fn_6802434Z8�5
.�+
!�
inputs����������
p 

 
� "�����������
%__inference_signature_wrapper_6802372�D�A
� 
:�7
5
RewardInput&�#
RewardInput����������"3�0
.
dense_38"�
dense_38���������