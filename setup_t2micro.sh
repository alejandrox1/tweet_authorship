#!/bin/bash
# set -x # Use for debug mode

# settings
export name="Ec2 Instance"
export cidr="0.0.0.0/0"
export instance_type="t2.micro"

# get the correct ami
export region=`aws configure get region`
if [ $region = "us-west-2" ]; then 
   export ami="ami-a58d0dc5"
else
  echo "Only us-west-2 (Oregon) is currently supported"
  exit 1
fi

export vpcId=`aws ec2 create-vpc --cidr-block 10.0.0.0/28 --query 'Vpc.VpcId' --output text`
aws ec2 create-tags --resources $vpcId --tags --tags Key=Name,Value=$name
aws ec2 modify-vpc-attribute --vpc-id $vpcId --enable-dns-support "{\"Value\":true}"
aws ec2 modify-vpc-attribute --vpc-id $vpcId --enable-dns-hostnames "{\"Value\":true}"

export internetGatewayId=`aws ec2 create-internet-gateway --query 'InternetGateway.InternetGatewayId' --output text`
aws ec2 create-tags --resources $internetGatewayId --tags --tags Key=Name,Value=$name-gateway
aws ec2 attach-internet-gateway --internet-gateway-id $internetGatewayId --vpc-id $vpcId

export subnetId=`aws ec2 create-subnet --vpc-id $vpcId --cidr-block 10.0.0.0/28 --query 'Subnet.SubnetId' --output text`
aws ec2 create-tags --resources $internetGatewayId --tags --tags Key=Name,Value=$name-subnet

export routeTableId=`aws ec2 create-route-table --vpc-id $vpcId --query 'RouteTable.RouteTableId' --output text`
aws ec2 create-tags --resources $routeTableId --tags --tags Key=Name,Value=$name-route-table
export routeTableAssoc=`aws ec2 associate-route-table --route-table-id $routeTableId --subnet-id $subnetId --output text`
aws ec2 create-route --route-table-id $routeTableId --destination-cidr-block 0.0.0.0/0 --gateway-id $internetGatewayId

export securityGroupId=`aws ec2 create-security-group --group-name $name-security-group --description "SG for fast.ai machine" --vpc-id $vpcId --query 'GroupId' --output text`
# ssh 
aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 22 --cidr $cidr
# jupyter notebook
aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 8888-8898 --cidr $cidr

if [ ! -d ~/.ssh ]
then
	mkdir ~/.ssh
fi

if [ ! -f ~/.ssh/aws-key-$name.pem ] 
then
	aws ec2 create-key-pair --key-name aws-key-$name --query 'KeyMaterial' --output text > ~/.ssh/aws-key-$name.pem
	chmod 400 ~/.ssh/aws-key-$name.pem
fi

export instanceId=`aws ec2 run-instances --image-id $ami --count 1 --instance-type $instance_type --key-name aws-key-$name --security-group-ids $securityGroupId --subnet-id $subnetId --associate-public-ip-address --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 128, \"VolumeType\": \"gp2\" } } ]" --query 'Instances[0].InstanceId' --output text`
aws ec2 create-tags --resources $instanceId --tags --tags Key=Name,Value=$name-gpu-machine
export allocAddr=`aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text`

echo Waiting for instance start...
aws ec2 wait instance-running --instance-ids $instanceId
sleep 10 # wait for ssh service to start running too
export assocId=`aws ec2 associate-address --instance-id $instanceId --allocation-id $allocAddr --query 'AssociationId' --output text`
export instanceUrl=`aws ec2 describe-instances --instance-ids $instanceId --query 'Reservations[0].Instances[0].PublicDnsName' --output text`
#export ebsVolume=`aws ec2 describe-instance-attribute --instance-id $instanceId --attribute  blockDeviceMapping  --query BlockDeviceMappings[0].Ebs.VolumeId --output text`

# reboot instance, because I was getting "Failed to initialize NVML: Driver/library version mismatch" 
# error when running the nvidia-smi command
# see also http://forums.fast.ai/t/no-cuda-capable-device-is-detected/168/13
aws ec2 reboot-instances --instance-ids $instanceId

# save commands to file 
echo \# Connect to your instance: > t2micro_commands.txt # overwrite existing file
echo ssh -i ~/.ssh/aws-key-$name.pem ubuntu@$instanceUrl >> t2micro_commands.txt
echo \# Stop your instance: : >> t2micro_commands.txt
echo aws ec2 stop-instances --instance-ids $instanceId  >> t2micro_commands.txt
echo \# Start your instance: >> t2micro_commands.txt
echo aws ec2 start-instances --instance-ids $instanceId  >> t2micro_commands.txt
echo Reboot your instance: >> t2micro_commands.txt
echo aws ec2 reboot-instances --instance-ids $instanceId  >> t2micro_commands.txt
echo "" 
# export vars to be sure
echo export instanceId=$instanceId >> t2micro_commands.txt
echo export subnetId=$subnetId >> t2micro_commands.txt
echo export securityGroupId=$securityGroupId >> t2micro_commands.txt
echo export instanceUrl=$instanceUrl >> t2micro_commands.txt
echo export routeTableId=$routeTableId >> t2micro_commands.txt
echo export name=$name >> t2micro_commands.txt
echo export vpcId=$vpcId >> t2micro_commands.txt
echo export internetGatewayId=$internetGatewayId >> t2micro_commands.txt
echo export subnetId=$subnetId >> t2micro_commands.txt
echo export allocAddr=$allocAddr >> t2micro_commands.txt
echo export assocId=$assocId >> t2micro_commands.txt
echo export routeTableAssoc=$routeTableAssoc >> t2micro_commands.txt

# save delete commands for cleanup
echo "#!/bin/bash" > t2micro_remove.sh # overwrite existing file
echo aws ec2 disassociate-address --association-id $assocId >> t2micro_remove.sh
echo aws ec2 release-address --allocation-id $allocAddr >> t2micro_remove.sh

# volume gets deleted with the instance automatically
echo aws ec2 terminate-instances --instance-ids $instanceId >> t2micro_remove.sh
echo aws ec2 wait instance-terminated --instance-ids $instanceId >> t2micro_remove.sh
echo aws ec2 delete-security-group --group-id $securityGroupId >> t2micro_remove.sh

echo aws ec2 disassociate-route-table --association-id $routeTableAssoc >> t2micro_remove.sh
echo aws ec2 delete-route-table --route-table-id $routeTableId >> t2micro_remove.sh

echo aws ec2 detach-internet-gateway --internet-gateway-id $internetGatewayId --vpc-id $vpcId >> t2micro_remove.sh
echo aws ec2 delete-internet-gateway --internet-gateway-id $internetGatewayId >> t2micro_remove.sh
echo aws ec2 delete-subnet --subnet-id $subnetId >> t2micro_remove.sh

echo aws ec2 delete-vpc --vpc-id $vpcId >> t2micro_remove.sh
echo echo If you want to delete the key-pair, please do it manually. >> t2micro_remove.sh

chmod +x t2micro_remove.sh

echo All done. Find all you need to connect in the t2micro_commands.txt file and to remove the stack call t2micro_remove.sh
echo Connect to your instance: ssh -i ~/.ssh/aws-key-$name.pem ubuntu@$instanceUrl
